import io
import asyncio
import logging
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HandTalkServer")

app = FastAPI(title="HandTalk PyTorch Server API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 100 VSL Gesture Labels
VSL_LABELS = [
    "An ủi", "Áp dụng", "Ăn", "Ăn mừng", "Ban ngày",
    "Ban đêm", "Bàn tay", "Băn khoăn", "Bạn thân", "Bế mạc",
    "Bệnh nhân", "Bệnh viện", "Biết", "Biếu tặng", "Bộ y tế",
    "Cá", "Cách ly", "Cám dỗ", "Cần", "Cảm ơn",
    "Chào", "Chân", "Chấp nhận", "Chạy", "Chậm lại",
    "Chiều", "Cho", "Chúng ta", "Con gấu", "Có thể",
    "Cơ thể", "Cứu", "Dạy dỗ", "Dễ", "Ghét",
    "Giúp", "Hâm mộ", "Ho", "Hôm nay", "Họ",
    "Học sinh", "Hy sinh", "Kết hôn", "Khai báo", "Khẩu trang",
    "Khóc", "Khu cách ly", "Lây bệnh", "Lo lắng", "Mời vào",
    "Mua", "Nặng", "Nghe", "Nghỉ ngơi", "Ngón tay",
    "Nhà", "Nhầm", "Nhìn", "Nhớ", "Nói",
    "Nói xấu", "Nôn ói", "Ô tô", "Phạt", "Phía sau",
    "Phỏng vấn", "Phục hồi", "Rau", "Rẽ phải", "Rẽ trái",
    "San sẻ", "Sốt", "Sử dụng", "Tập luyện", "Thất lạc",
    "Thăm", "Thích", "Thương", "Thức ăn", "Thức dậy",
    "Tôi", "Tối", "Trưa", "Trường học", "Uống",
    "Ủng hộ", "Vâng lời", "Virus", "Xa", "Xe máy",
    "Xe đạp", "Xin lỗi", "Xin phép", "Xuất viện", "Xúc động",
    "Đâu", "Đầu", "Đẹp", "Đi", "Đồng ý"
]

REQUIRED_FRAMES = 16


class PyTorchModelWrapper:
    """
    Wrapper for your PyTorch VideoMAE / ResNet Model.
    Replace the code below with your model loading and inference logic:
    
        import torch
        self.model = torch.load('path_to_model.pth')
        self.model.eval()
    """

    def __init__(self):
        logger.info("PyTorch Model Initialized")

    def predict(self, frame_sequence: List[Image.Image]) -> Dict:
        """
        Runs PyTorch inference on a sequence of PIL images (16 frames).
        """
        if len(frame_sequence) < REQUIRED_FRAMES:
            return None

        # TODO: Replace with your PyTorch inference:
        # inputs = preprocess(frame_sequence)
        # outputs = self.model(inputs)
        # class_idx = outputs.argmax().item()
        
        # Default placeholder demo response
        class_idx = 20  # "Chào"
        confidence = 0.96

        return {
            "class_index": class_idx,
            "label": VSL_LABELS[class_idx],
            "confidence": confidence
        }


model_runner = PyTorchModelWrapper()


@app.get("/")
def read_root():
    return {"status": "ok", "message": "HandTalk PyTorch WebSocket Server API Running"}


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to WebSocket stream")
    frame_buffer: List[Image.Image] = []

    try:
        while True:
            # Receive binary JPEG bytes from Android client
            data = await websocket.receive_bytes()

            try:
                image = Image.open(io.BytesIO(data)).convert("RGB")
                frame_buffer.append(image)

                if len(frame_buffer) > REQUIRED_FRAMES:
                    frame_buffer.pop(0)

                if len(frame_buffer) == REQUIRED_FRAMES:
                    result = model_runner.predict(frame_buffer)
                    if result:
                        await websocket.send_json(result)

            except Exception as e:
                logger.error(f"Error processing frame: {e}")

    except WebSocketDisconnect:
        logger.info("Client disconnected")
