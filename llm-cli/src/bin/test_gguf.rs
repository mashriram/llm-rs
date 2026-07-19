fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "models/gemma4_tokenizer.json".to_string());
    let tokenizer = tokenizers::Tokenizer::from_file(&path).unwrap();
    let ids = vec![71499, 563, 886, 529, 506, 2390, 11001, 8290, 529, 4135, 568, 51528, 607, 129210, 1929, 236764, 506, 3188, 8825, 4912, 236764, 532, 506, 7209, 8825, 4912, 769, 799, 506, 33055, 3755, 236764, 5213, 50918, 563, 506, 4912, 529, 29433, 600, 7519, 1534, 1027, 1156, 7751, 600, 735, 3413, 653, 2778];
    println!("Decoded: {:?}", tokenizer.decode(&ids, false).unwrap());
    Ok(())
}
