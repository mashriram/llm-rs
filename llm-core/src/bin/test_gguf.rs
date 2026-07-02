fn main() -> anyhow::Result<()> {
    let tok = llm_core::tokenizer::LlmTokenizer::from_file("/home/mukundan/learning/llm/gemma4_tokenizer.json")?;
    let ids = vec![2, 235303, 145, 235292, 108, 2307, 603, 15138, 235336, 142, 108, 235303, 145, 235292];
    println!("Decoded: {:?}", tok.decode(&ids, false)?);
    for id in &ids {
        println!("  {}: {:?}", id, tok.decode(&[*id], false)?);
    }
    Ok(())
}
