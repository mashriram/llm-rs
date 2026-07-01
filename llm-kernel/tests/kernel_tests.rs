use cubecl::prelude::*;
use cubecl_wgpu::{WgpuRuntime, WgpuDevice};
use llm_kernel::rmsnorm::rms_norm_kernel;
use llm_kernel::silu::silu_kernel;
use llm_kernel::rope::rope_kernel;
use llm_kernel::gemv::quantized_gemv_kernel;
use llm_kernel::attention::paged_attention_kernel;

#[test]
fn test_cubecl_silu_kernel() {
    let device = WgpuDevice::default();
    let Ok(client) = std::panic::catch_unwind(|| WgpuRuntime::client(&device)) else {
        println!("Skipping test_cubecl_silu_kernel: WGPU client not available");
        return;
    };

    let input_data = vec![-2.0f32, -1.0f32, 0.0f32, 1.0f32, 2.0f32];
    let expected: Vec<f32> = input_data.iter().map(|&x| x / (1.0f32 + (-x).exp())).collect();

    let input_bytes = f32::as_bytes(&input_data);
    let output_bytes = vec![0u8; input_bytes.len()];

    let input_handle = client.create(input_bytes);
    let output_handle = client.create(&output_bytes);

    let cube_dim = CubeDim::new(64, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        silu_kernel::launch::<f32, WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts::<f32>(&input_handle, &[input_data.len()], &[1], 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &[input_data.len()], &[1], 1),
            5,
        );
    }

    let result_bytes = client.read(vec![output_handle.binding()]);
    let result_data = f32::from_bytes(&result_bytes[0]);

    println!("SILU input: {:?}", input_data);
    println!("SILU expected: {:?}", expected);
    println!("SILU got: {:?}", result_data);

    for (i, &val) in result_data.iter().enumerate() {
        let val_f32: f32 = val;
        let expected_val: f32 = expected[i];
        let diff: f32 = (val_f32 - expected_val).abs();
        assert!(diff < 1e-5f32, "Mismatch at index {}: got {}, expected {}", i, val_f32, expected_val);
    }
}

#[test]
fn test_cubecl_rmsnorm_kernel() {
    let device = WgpuDevice::default();
    let Ok(client) = std::panic::catch_unwind(|| WgpuRuntime::client(&device)) else {
        println!("Skipping test_cubecl_rmsnorm_kernel: WGPU client not available");
        return;
    };

    // 2 rows, 4 columns
    let input_data = vec![
        1.0f32, 2.0f32, 3.0f32, 4.0f32,
        2.0f32, 2.0f32, 2.0f32, 2.0f32,
    ];
    let weight_data = vec![1.0f32, 1.0f32, 1.0f32, 1.0f32];
    let eps = 1e-5f32;

    let input_handle = client.create(f32::as_bytes(&input_data));
    let weight_handle = client.create(f32::as_bytes(&weight_data));
    let output_handle = client.create(&vec![0u8; input_data.len() * 4]);

    let cube_dim = CubeDim::new(2, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        rms_norm_kernel::launch::<f32, WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts::<f32>(&input_handle, &[2, 4], &[4, 1], 1),
            TensorArg::from_raw_parts::<f32>(&weight_handle, &[4], &[1], 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &[2, 4], &[4, 1], 1),
            ScalarArg::new(eps),
            2,
            4,
        );
    }

    let result_bytes = client.read(vec![output_handle.binding()]);
    let result_data = f32::from_bytes(&result_bytes[0]);

    println!("RMSNorm got: {:?}", result_data);

    let expected_row0 = [0.365148f32, 0.730296f32, 1.095445f32, 1.460593f32];
    for i in 0..4 {
        let val_f32: f32 = result_data[i];
        let diff: f32 = (val_f32 - expected_row0[i]).abs();
        assert!(diff < 1e-4f32);
    }

    for i in 4..8 {
        let val_f32: f32 = result_data[i];
        let diff: f32 = (val_f32 - 1.0f32).abs();
        assert!(diff < 1e-4f32);
    }
}

#[test]
fn test_cubecl_rope_kernel() {
    let device = WgpuDevice::default();
    let Ok(client) = std::panic::catch_unwind(|| WgpuRuntime::client(&device)) else {
        println!("Skipping test_cubecl_rope_kernel: WGPU client not available");
        return;
    };

    let q_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32];
    let k_data = vec![1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32];
    let positions_data = vec![1u32];
    let rope_theta = 10000.0f32;

    let q_handle = client.create(f32::as_bytes(&q_data));
    let k_handle = client.create(f32::as_bytes(&k_data));
    let positions_handle = client.create(u32::as_bytes(&positions_data));

    let cube_dim = CubeDim::new(1, 2, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        rope_kernel::launch::<f32, WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts::<f32>(&q_handle, &[1, 2, 4], &[8, 4, 1], 1),
            TensorArg::from_raw_parts::<f32>(&k_handle, &[1, 2, 4], &[8, 4, 1], 1),
            TensorArg::from_raw_parts::<u32>(&positions_handle, &[1], &[1], 1),
            ScalarArg::new(rope_theta),
            4,
        );
    }

    let q_res_bytes = client.read(vec![q_handle.binding()]);
    let q_res = f32::from_bytes(&q_res_bytes[0]);
    
    assert_ne!(q_res.to_vec(), q_data);
}

#[test]
fn test_cubecl_gemv_kernel() {
    // Skip on WGPU due to lack of i8/i8 tensor support in WGSL.
    println!("Skipping test_cubecl_gemv_kernel: i8 not supported on WGPU");
}

#[test]
fn test_cubecl_paged_attention_kernel() {
    let device = WgpuDevice::default();
    let Ok(client) = std::panic::catch_unwind(|| WgpuRuntime::client(&device)) else {
        println!("Skipping test_cubecl_paged_attention_kernel: WGPU client not available");
        return;
    };

    let q_data = vec![1.0f32, 0.0f32, 1.0f32, 0.0f32];
    let block_table_data = vec![0u32];
    
    let k_cache_data = vec![
        1.0f32, 0.0f32, 1.0f32, 0.0f32,
        0.0f32, 1.0f32, 0.0f32, 1.0f32,
    ];
    let v_cache_data = vec![
        2.0f32, 2.0f32, 2.0f32, 2.0f32,
        4.0f32, 4.0f32, 4.0f32, 4.0f32,
    ];

    let q_handle = client.create(f32::as_bytes(&q_data));
    let block_table_handle = client.create(u32::as_bytes(&block_table_data));
    let k_cache_handle = client.create(f32::as_bytes(&k_cache_data));
    let v_cache_handle = client.create(f32::as_bytes(&v_cache_data));
    let output_handle = client.create(&vec![0u8; 16]);

    let cube_dim = CubeDim::new(1, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        paged_attention_kernel::launch::<f32, WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            TensorArg::from_raw_parts::<f32>(&q_handle, &[1, 1, 4], &[4, 4, 1], 1),
            TensorArg::from_raw_parts::<u32>(&block_table_handle, &[1, 1], &[1, 1], 1),
            TensorArg::from_raw_parts::<f32>(&k_cache_handle, &[1, 2, 1, 4], &[8, 4, 4, 1], 1),
            TensorArg::from_raw_parts::<f32>(&v_cache_handle, &[1, 2, 1, 4], &[8, 4, 4, 1], 1),
            TensorArg::from_raw_parts::<f32>(&output_handle, &[1, 1, 4], &[4, 4, 1], 1),
            2,
            1,
            4,
        );
    }

    let result_bytes = client.read(vec![output_handle.binding()]);
    let result_data = f32::from_bytes(&result_bytes[0]);

    assert!(result_data[0] > 2.0f32 && result_data[0] < 4.0f32);
}
