use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::{ws::{WebSocket, WebSocketUpgrade}, Multipart},
    response::IntoResponse,
};
use serde::{Serialize, Deserialize};
use futures::{sink::SinkExt, stream::StreamExt};
use std::sync::Arc;
use tch::{nn, vision::{imagenet, resnet}, Device};
use image::ImageFormat;
use std::io::Cursor;

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    version: String,
}

#[derive(Deserialize, Serialize)]
struct DetectionResult {
    class: String,
    confidence: f32,
}

async fn status() -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn process_image(base64_image: &str) -> Vec<DetectionResult> {
    println!("Processing image: {}", base64_image);

    vec![DetectionResult {
        class: "fish".to_string(),
        confidence: 1.0,
    }]
}

async fn handle_socket(mut socket: WebSocket) {
    while let Some(msg) = socket.recv().await {
        let msg = if let Ok(msg) = msg {
            msg
        } else {
            return;
        };

        if let Ok(text) = msg.to_text() {
            if text == "ping" {
                if socket.send("pong".into()).await.is_err() {
                    return;
                }
                continue;
            }

            // Process image if it's base64 encoded
            if text.starts_with("data:image") {
                let base64_image = text.split(",").nth(1).unwrap_or("");
                let results = process_image(base64_image).await;
                
                if let Ok(json) = serde_json::to_string(&results) {
                    if socket.send(json.into()).await.is_err() {
                        return;
                    }
                }
            }
        }
    }
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/status", get(status))
        .route("/ws", get(ws_handler));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Server running on http://0.0.0.0:3000");
    
    axum::serve(listener, app).await.unwrap();
}
