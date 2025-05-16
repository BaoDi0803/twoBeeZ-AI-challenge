print("==> Đang chạy file detect_and_price.py 22")

from ultralytics import YOLO
from pathlib import Path
import json
from collections import Counter

def main():
    # Load mô hình đã train
    model = YOLO('runs/train/canteen8/weights/best.pt')

    # Load ảnh đầu vào
    img_path = "test.jpg"
    img = str(Path(img_path))

    # Load menu và giá tiền
    with open('menu.json', 'r', encoding='utf-8') as f:
        menu = json.load(f)

    # Nhận diện
    results = model(img)[0]
    predictions = results.boxes.data
    names = model.names

    # Đếm số lượng từng món
    labels = [names[int(cls)] for *_, conf, cls in predictions]
    counts = Counter(labels)

    # Hiển thị kết quả và tính tiền
    print("\n==> Kết quả nhận diện:")
    total = 0
    for item, count in counts.items():
        price = menu.get(item, 0)
        subtotal = price * count
        total += subtotal
        print(f" - {item}: {count} x {price} = {subtotal} đ")

    print(f"\n==> Tổng cộng: {total} đ")

if __name__ == '__main__':
    main()
