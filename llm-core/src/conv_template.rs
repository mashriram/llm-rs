use std::collections::HashMap;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<HashMap<String, String>>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

// Custom deserializer for messages list which can be [["role", "content"], ...] or similar
struct MessagesVisitor;

impl<'de> serde::de::Visitor<'de> for MessagesVisitor {
    type Value = Vec<Message>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a list of messages where each message is a [role, content] pair")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut messages = Vec::new();
        while let Some(raw_msg) = seq.next_element::<serde_json::Value>()? {
            if let Some(arr) = raw_msg.as_array() {
                if arr.len() == 2 {
                    let role = arr[0].as_str().unwrap_or("").to_string();
                    let content = if let Some(text) = arr[1].as_str() {
                        MessageContent::Text(text.to_string())
                    } else {
                        let parts: Vec<HashMap<String, String>> = serde_json::from_value(arr[1].clone())
                            .map_err(serde::de::Error::custom)?;
                        MessageContent::Parts(parts)
                    };
                    messages.push(Message { role, content });
                }
            }
        }
        Ok(messages)
    }
}

fn deserialize_messages<'de, D>(deserializer: D) -> Result<Vec<Message>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserializer.deserialize_seq(MessagesVisitor)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub name: String,
    pub system_template: String,
    pub system_message: String,
    pub roles: HashMap<String, String>,
    pub role_templates: HashMap<String, String>,
    #[serde(deserialize_with = "deserialize_messages")]
    pub messages: Vec<Message>,
    pub seps: Vec<String>,
    pub role_content_sep: String,
    pub role_empty_sep: String,
    pub stop_str: Vec<String>,
    pub add_role_after_system_message: bool,
    pub stop_token_ids: Vec<u32>,
}

impl Conversation {
    pub fn from_json(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str).context("Failed to parse Conversation from JSON")
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(Message {
            role: role.to_string(),
            content: MessageContent::Text(content.to_string()),
        });
    }

    pub fn render_prompt(&self) -> String {
        // Start with the system prompt formatted with the system template
        let mut prompt = self.system_template.replace("{system_message}", &self.system_message);

        for msg in &self.messages {
            let role_name = self.roles.get(&msg.role).cloned().unwrap_or_else(|| msg.role.clone());
            let content_str = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Parts(parts) => {
                    let mut text = String::new();
                    for part in parts {
                        if let Some(t) = part.get("text") {
                            text.push_str(t);
                        }
                    }
                    text
                }
            };
            
            let role_tmpl = self.role_templates.get(&msg.role).cloned()
                .unwrap_or_else(|| "{anchor}".to_string());
            
            let formatted_msg = role_tmpl
                .replace(&format!("{{{}_message}}", msg.role), &content_str)
                .replace(&format!("{{{}}}", msg.role), &content_str);

            prompt.push_str(&role_name);
            prompt.push_str(&self.role_content_sep);
            prompt.push_str(&formatted_msg);
            
            if !self.seps.is_empty() {
                prompt.push_str(&self.seps[0]);
            }
        }
        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_template_load_json_text_content() {
        let conv_template = r#"{
            "name": "test",
            "system_template": "abc{system_message}",
            "system_message": "de",
            "roles": {
              "user": "Instruct",
              "assistant": "Output",
              "tool": "Instruct"
            },
            "role_templates": {
              "user": "{user_message}",
              "assistant": "{assistant_message}",
              "tool": "{tool_message}"
            },
            "messages": [["Instruct", "Hello"], ["Output", "Hey"]],
            "seps": [
              "\n"
            ],
            "role_content_sep": ": ",
            "role_empty_sep": ":",
            "stop_str": [
              "<|endoftext|>"
            ],
            "add_role_after_system_message": false,
            "stop_token_ids": [
              50256
            ]
        }"#;

        let conv = Conversation::from_json(conv_template).unwrap();
        assert_eq!(conv.name, "test");
        assert_eq!(conv.system_template, "abc{system_message}");
        assert_eq!(conv.system_message, "de");
        assert_eq!(conv.roles.get("user").unwrap(), "Instruct");
        assert_eq!(conv.roles.get("assistant").unwrap(), "Output");
        assert_eq!(conv.roles.get("tool").unwrap(), "Instruct");
        assert_eq!(conv.role_templates.get("user").unwrap(), "{user_message}");
        assert_eq!(conv.role_templates.get("assistant").unwrap(), "{assistant_message}");
        assert_eq!(conv.role_templates.get("tool").unwrap(), "{tool_message}");
        
        assert_eq!(conv.messages[0].role, "Instruct");
        match &conv.messages[0].content {
            MessageContent::Text(text) => assert_eq!(text, "Hello"),
            _ => panic!("Expected text content"),
        }
        
        assert_eq!(conv.messages[1].role, "Output");
        match &conv.messages[1].content {
            MessageContent::Text(text) => assert_eq!(text, "Hey"),
            _ => panic!("Expected text content"),
        }

        assert_eq!(conv.seps[0], "\n");
        assert_eq!(conv.role_content_sep, ": ");
        assert_eq!(conv.role_empty_sep, ":");
        assert_eq!(conv.stop_str[0], "<|endoftext|>");
        assert_eq!(conv.add_role_after_system_message, false);
        assert_eq!(conv.stop_token_ids[0], 50256);
    }

    #[test]
    fn test_conv_template_load_json_parts_content() {
        let conv_template = r#"{
            "name": "test",
            "system_template": "abc{system_message}",
            "system_message": "de",
            "roles": {
              "user": "Instruct",
              "assistant": "Output",
              "tool": "Instruct"
            },
            "role_templates": {
              "user": "{user_message}",
              "assistant": "{assistant_message}",
              "tool": "{tool_message}"
            },
            "messages": [["Instruct", [
              {"type": "text", "text": "What's in the image?"},
              {"type": "image_url", "image_url": "https://example.com/image.jpg"}
            ]]],
            "seps": [
              "\n"
            ],
            "role_content_sep": ": ",
            "role_empty_sep": ":",
            "stop_str": [
              "<|endoftext|>"
            ],
            "add_role_after_system_message": false,
            "stop_token_ids": [
              50256
            ]
        }"#;

        let conv = Conversation::from_json(conv_template).unwrap();
        assert_eq!(conv.name, "test");
        assert_eq!(conv.messages[0].role, "Instruct");
        
        match &conv.messages[0].content {
            MessageContent::Parts(parts) => {
                assert_eq!(parts[0].get("type").unwrap(), "text");
                assert_eq!(parts[0].get("text").unwrap(), "What's in the image?");
                assert_eq!(parts[1].get("type").unwrap(), "image_url");
                assert_eq!(parts[1].get("image_url").unwrap(), "https://example.com/image.jpg");
            }
            _ => panic!("Expected parts content"),
        }
    }
}
