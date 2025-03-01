use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        Multipart,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use image::ImageFormat;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::sync::Arc;
use tch::{
    nn,
    vision::{imagenet, resnet},
    Device,
};

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    version: String,
}

#[derive(Deserialize, Debug)]
struct RoboflowPrediction {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    confidence: f32,
    class: String,
    class_id: u32,
}

#[derive(Deserialize, Debug)]
struct RoboflowResponse {
    predictions: Vec<RoboflowPrediction>,
}

#[derive(Serialize, Deserialize, Debug)]
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
    let api_key = std::env::var("ROBOFLOW_API_KEY").unwrap();
    let model_id = std::env::var("ROBOFLOW_MODEL_ID").unwrap();
    let model_version = std::env::var("ROBOFLOW_MODEL_VERSION").unwrap();
    let url = format!(
        "https://detect.roboflow.com/{}/{}?api_key={}",
        model_id, model_version, api_key
    );

    let client = Client::new();
    let response = client
        .post(&url)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(format!("image={}", base64_image)) // Correctly formatting the body
        .send()
        .await;

    match response {
        Ok(resp) if resp.status().is_success() => match resp.json::<RoboflowResponse>().await {
            Ok(json_response) => json_response
                .predictions
                .into_iter()
                .map(|p| DetectionResult {
                    class: p.class,
                    confidence: p.confidence,
                })
                .collect(),
            Err(_) => {
                eprintln!("Failed to parse Roboflow response.");
                vec![]
            }
        },
        Ok(resp) => {
            let error_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            eprintln!("Roboflow API error: {}", error_text);
            vec![]
        }
        Err(err) => {
            eprintln!("Failed to send request: {:?}", err);
            vec![]
        }
    }
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
