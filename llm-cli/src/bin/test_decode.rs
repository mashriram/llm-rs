fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "models/gemma4_tokenizer.json".to_string());
    let tokenizer = tokenizers::Tokenizer::from_file(&path).unwrap();
    let ids = vec![12766, 3140, 3140, 248712, 194983, 194983, 194983, 194983, 194983, 194983, 232325, 232325];
    for id in ids {
        let s = tokenizer.decode(&[id], false).unwrap();
        println!("{}: {:?}", id, s);
    }
    Ok(())
}
