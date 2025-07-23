#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::doc_markdown)]
#![doc = include_str!("../README.md")]

use reqwest::{
    header::{self, HeaderMap, HeaderValue},
    Client as Http, StatusCode,
};
use serde_json::json;
use std::env;
use types::{Error, Include, InputItemList, Request, Response, ResponseResult};
#[cfg(feature = "stream")]
use {
    async_fn_stream::try_fn_stream,
    futures::{Stream, StreamExt},
    reqwest_eventsource::{Event as EventSourceEvent, RequestBuilderExt},
    types::Event,
};

/// Types for interacting with the Responses API.
pub mod types;

/// Re-export the derive macro so users only need `openai_responses` as a dependency.
pub use openai_responses_derive::OpenAiFunction;

/// The OpenAI Responses API Client.
#[derive(Debug, Clone)]
pub struct Client {
    client: reqwest::Client,
    base_url: String,
}

/// Errors that can occur when creating a new Client.
#[derive(Debug, thiserror::Error)]
pub enum CreateError {
    /// The provided API key contains invalid header value characters. Only visible ASCII characters (32-127) are permitted.
    #[error(
        "The provided API key contains invalid header value characters. Only visible ASCII characters (32-127) are permitted."
    )]
    InvalidApiKey,
    /// Failed to create the HTTP Client
    #[error("Failed to create the HTTP Client: {0}")]
    CouldNotCreateClient(#[from] reqwest::Error),
    /// Could not retrieve the ``OPENAI_API_KEY`` env var
    #[error("Could not retrieve the $OPENAI_API_KEY env var")]
    ApiKeyNotFound,
}

#[cfg(feature = "stream")]
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("{0}")]
    Stream(#[from] reqwest_eventsource::Error),
    #[error("Failed to parse event data: {0}")]
    Parsing(#[from] serde_json::Error),
}

impl Client {
    /// Creates a new Client with the given API key.
    ///
    /// # Errors
    /// - `CreateError::CouldNotCreateClient` if the HTTP Client could not be created.
    /// - `CreateError::InvalidApiKey` if the API key contains invalid header value characters.
    pub fn new(api_key: &str) -> Result<Self, CreateError> {
        let client = Http::builder()
            .default_headers(HeaderMap::from_iter([(
                header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {api_key}"))
                    .map_err(|_| CreateError::InvalidApiKey)?,
            )]))
            .build()?;

        Ok(Self {
            client,
            base_url: "https://api.openai.com".to_owned(),
        })
    }

    /// Set the base URL for the OpenAI API.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Creates a new Client from the `OPENAI_API_KEY` environment variable.
    ///
    /// # Errors
    /// - `CreateError::CouldNotCreateClient` if the HTTP Client could not be created.
    /// - `CreateError::InvalidApiKey` if the API key contains invalid header value characters.
    /// - `CreateError::ApiKeyNotFound` if the `OPENAI_API_KEY` environment variable is not set or contains an equal sign or NUL (`'='` or `'\0'`).
    pub fn from_env() -> Result<Self, CreateError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| CreateError::ApiKeyNotFound)?;

        Self::new(&api_key)
    }

    /// Creates a model response.
    ///
    /// Provide [text](https://platform.openai.com/docs/guides/text) or [image](https://platform.openai.com/docs/guides/images) inputs to generate [text](https://platform.openai.com/docs/guides/text) or [JSON](https://platform.openai.com/docs/guides/structured-outputs) outputs.
    /// Have the model call your own [custom code](https://platform.openai.com/docs/guides/function-calling) or use built-in [tools](https://platform.openai.com/docs/guides/tools) like [web search](https://platform.openai.com/docs/guides/tools-web-search) or [file search](https://platform.openai.com/docs/guides/tools-file-search) to use your own data as input for the model's response.
    /// To receive a stream of tokens as they are generated, use the `stream` function instead.
    ///
    /// ## Errors
    ///
    /// Errors if the request fails to send or has a non-200 status code (except for 400, which will return an OpenAI error instead).
    pub async fn create(
        &self,
        mut request: Request,
    ) -> Result<Result<Response, Error>, reqwest::Error> {
        // Use the `stream` function to stream the response.
        request.stream = Some(false);

        let mut response = self
            .client
            .post(format!("{}/v1/responses", self.base_url))
            .json(&request)
            .send()
            .await?;

        if response.status() != StatusCode::BAD_REQUEST {
            response = response.error_for_status()?;
        }

        response.json::<ResponseResult>().await.map(Into::into)
    }

    #[cfg(feature = "stream")]
    /// Creates a model response and streams it back as it is generated.
    ///
    /// Provide [text](https://platform.openai.com/docs/guides/text) or [image](https://platform.openai.com/docs/guides/images) inputs to generate [text](https://platform.openai.com/docs/guides/text) or [JSON](https://platform.openai.com/docs/guides/structured-outputs) outputs.
    /// Have the model call your own [custom code](https://platform.openai.com/docs/guides/function-calling) or use built-in [tools](https://platform.openai.com/docs/guides/tools) like [web search](https://platform.openai.com/docs/guides/tools-web-search) or [file search](https://platform.openai.com/docs/guides/tools-file-search) to use your own data as input for the model's response.
    ///
    /// To receive the response as a regular HTTP response, use the `create` function.
    pub fn stream(
        &self,
        mut request: Request,
    ) -> impl Stream<Item = Result<Event, StreamError>> + Send + 'static + use<'_> {
        // Use the `create` function to receive a regular HTTP response.
        request.stream = Some(true);

        let mut event_source = self
            .client
            .post(format!("{}/v1/responses", self.base_url))
            .json(&request)
            .eventsource()
            .unwrap_or_else(|_| unreachable!("Body is never a stream"));

        let stream = try_fn_stream(|emitter| async move {
            while let Some(event) = event_source.next().await {
                let message = match event {
                    Ok(EventSourceEvent::Open) => continue,
                    Ok(EventSourceEvent::Message(message)) => message,
                    Err(e @ reqwest_eventsource::Error::Transport(_)) => {
                        emitter.emit_err(StreamError::Stream(e)).await;
                        break;   // stop consuming – the EventSource will be dropped
                    }
                    Err(error) => {
                        if matches!(error, reqwest_eventsource::Error::StreamEnded) {
                            break;
                        }

                        emitter.emit_err(StreamError::Stream(error)).await;
                        continue;
                    }
                };

                match serde_json::from_str::<Event>(&message.data) {
                    Ok(event) => emitter.emit(event).await,
                    Err(error) => emitter.emit_err(StreamError::Parsing(error)).await,
                }
            }

            Ok(())
        });

        Box::pin(stream)
    }

    /// Retrieves a model response with the given ID.
    ///
    /// ## Errors
    ///
    /// Errors if the request fails to send or has a non-200 status code (except for 400, which will return an OpenAI error instead).
    pub async fn get(
        &self,
        response_id: &str,
        include: Option<Include>,
    ) -> Result<Result<Response, Error>, reqwest::Error> {
        let mut response = self
            .client
            .get(format!("{}/v1/responses/{response_id}", self.base_url))
            .query(&json!({ "include": include }))
            .send()
            .await?;

        if response.status() != StatusCode::BAD_REQUEST {
            response = response.error_for_status()?;
        }

        response.json::<ResponseResult>().await.map(Into::into)
    }

    /// Deletes a model response with the given ID.
    ///
    /// ## Errors
    ///
    /// Errors if the request fails to send or has a non-200 status code.
    pub async fn delete(&self, response_id: &str) -> Result<(), reqwest::Error> {
        self.client
            .delete(format!("{}/v1/responses/{response_id}", self.base_url))
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }

    /// Returns a list of input items for a given response.
    ///
    /// ## Errors
    ///
    /// Errors if the request fails to send or has a non-200 status code.
    pub async fn list_inputs(&self, response_id: &str) -> Result<InputItemList, reqwest::Error> {
        self.client
            .get(format!(
                "{}/v1/responses/{response_id}/inputs",
                self.base_url
            ))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await
    }
}

/* -------------------------------------------------------------------------
   Helper: build `Tool` specs automatically from Rust types
   --------------------------------------------------------------------- */

#[cfg(feature = "schema")]
pub mod tool_builder {
    //! Helpers to generate an OpenAI *Tool* definition from a Rust structure.
    //!
    //! The implementation relies on the `schemars` crate to obtain a JSON
    //! Schema for the Rust type.  If you use the `OpenAiFunction` derive macro
    //! (`openai_responses_derive`) the macro will inject the necessary `NAME`
    //! and `DESCRIPTION` constants automatically – otherwise you can implement
    //! the [`IntoFunction`] trait manually.

    use schemars::JsonSchema;
    use serde_json::Value;

    use crate::types::Function;

    /// Recursively processes a JSON schema to replace oneOf with anyOf and add additionalProperties: false to all objects
    fn process_schema_recursively(value: &mut Value) {
        match value {
            Value::Object(map) => {
                // Replace oneOf with anyOf
                if let Some(one_of) = map.remove("oneOf") {
                    map.insert("anyOf".to_string(), one_of);
                }

                // Add additionalProperties: false to all object types
                if map.contains_key("type") {
                    if let Some(Value::String(type_str)) = map.get("type") {
                        if type_str == "object" {
                            map.entry("additionalProperties".to_string())
                                .or_insert_with(|| Value::Bool(false));
                        }
                    }
                }

                // Also handle objects that don't have explicit type but have properties
                if map.contains_key("properties") && !map.contains_key("type") {
                    map.entry("additionalProperties".to_string())
                        .or_insert_with(|| Value::Bool(false));
                }

                // Recursively process all values in the object
                for (_, v) in map.iter_mut() {
                    process_schema_recursively(v);
                }
            }
            Value::Array(arr) => {
                // Recursively process all items in arrays
                for item in arr.iter_mut() {
                    process_schema_recursively(item);
                }
            }
            _ => {
                // For primitive values, no processing needed
            }
        }
    }

    /// Trait implemented for structures that can be exposed to the model as a
    /// callable *tool* (i.e. function).
    ///
    /// The default `as_tool` implementation turns the [`schemars`] schema into
    /// the `parameters` object expected by OpenAI.
    pub trait IntoFunction: JsonSchema {
        /// Name under which the model should call the tool.
        const NAME: &'static str;
        /// Optional human-readable description.
        const DESCRIPTION: &'static str = "";

        /// Convert the Rust type into a [`Function`] definition.
        fn convert_into_function() -> Function {
            // Build full JSON-schema for the type
            let schema = schemars::schema_for!(Self);
            let mut parameters: Value =
                serde_json::to_value(&schema).expect("serialising JSON schema failed");

            // Strip keys that are not wanted by the OpenAI spec and process schema
            if let Value::Object(ref mut map) = parameters {
                map.remove("$schema");
                map.remove("title");
                // Ensure `additionalProperties` is always present (default: false)
                map.entry("additionalProperties")
                    .or_insert_with(|| Value::Bool(false));
            }

            // Recursively process the schema to replace oneOf with anyOf and add additionalProperties: false
            process_schema_recursively(&mut parameters);

            Function {
                name: Self::NAME.to_string(),
                description: if Self::DESCRIPTION.is_empty() {
                    None
                } else {
                    Some(Self::DESCRIPTION.to_string())
                },
                parameters,
                strict: true,
            }
        }
    }

    /// Convenience helper – returns the [`Function`]
    #[inline]
    pub fn tool<T: IntoFunction>() -> Function {
        T::convert_into_function()
    }
}
