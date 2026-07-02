fn main() -> anyhow::Result<()> {
    let tokenizer = tokenizers::Tokenizer::from_file("/home/mukundan/learning/llm/gemma4_tokenizer.json").unwrap();
    let ids = vec![12766, 3140, 3140, 248712, 194983, 194983, 194983, 194983, 194983, 194983, 232325, 232325];
    for id in ids {
        let s = tokenizer.decode(&[id], false).unwrap();
        println!("{}: {:?}", id, s);
    }
    Ok(())
}
