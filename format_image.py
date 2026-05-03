import os
import glob
from PIL import Image
from pathlib import Path


def format_image(path, image_format):
    """
    Chuyển đổi tất cả các ảnh trong thư mục path sang định dạng image_format.
    
    Args:
        path (str): Đường dẫn đến thư mục chứa ảnh
        image_format (str): Định dạng ảnh mong muốn (jpg, png, jpeg, bmp, etc.)
    
    Returns:
        dict: Thống kê số lượng ảnh đã chuyển đổi
    """
    # Chuẩn hóa định dạng (loại bỏ dấu chấm nếu có và chuyển thành chữ thường)
    image_format = image_format.lower().strip('.')
    
    # Danh sách các định dạng ảnh phổ biến
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(path):
        print(f"❌ Thư mục {path} không tồn tại!")
        return {"error": "Directory not found"}
    
    # Thống kê
    stats = {
        "total_found": 0,
        "converted": 0,
        "skipped": 0,
        "errors": 0
    }
    
    # Lấy tất cả các file ảnh
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(path, ext)))
        # Tìm cả trong các thư mục con
        image_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
    
    # Loại bỏ các file trùng lặp
    image_files = list(set(image_files))
    stats["total_found"] = len(image_files)
    
    print(f"📂 Tìm thấy {len(image_files)} ảnh trong {path}")
    print(f"🎯 Định dạng mục tiêu: .{image_format}")
    
    for img_path in image_files:
        try:
            # Lấy thông tin file
            file_path = Path(img_path)
            current_ext = file_path.suffix.lower().strip('.')
            
            # Kiểm tra xem có cần chuyển đổi không
            # Coi jpg và jpeg là giống nhau
            if (current_ext == image_format or 
                (current_ext in ['jpg', 'jpeg'] and image_format in ['jpg', 'jpeg'])):
                stats["skipped"] += 1
                continue
            
            # Đọc ảnh
            img = Image.open(img_path)
            
            # Chuyển đổi sang RGB nếu cần (đặc biệt cho JPEG)
            if image_format in ['jpg', 'jpeg'] and img.mode in ['RGBA', 'P', 'LA']:
                # Tạo background trắng
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ['RGBA', 'LA'] else None)
                img = rgb_img
            
            # Tạo tên file mới
            new_filename = file_path.stem + '.' + image_format
            new_path = file_path.parent / new_filename
            
            # Lưu ảnh với định dạng mới
            if image_format in ['jpg', 'jpeg']:
                img.save(new_path, 'JPEG', quality=95)
            elif image_format == 'png':
                img.save(new_path, 'PNG', optimize=True)
            else:
                img.save(new_path, image_format.upper())
            
            # Xóa file cũ nếu tên file khác nhau
            if img_path != str(new_path):
                os.remove(img_path)
                print(f"✅ Chuyển đổi: {file_path.name} → {new_filename}")
            
            stats["converted"] += 1
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {img_path}: {e}")
            stats["errors"] += 1
    
    # In thống kê
    print(f"\n{'='*60}")
    print(f"📊 KẾT QUẢ CHUYỂN ĐỔI:")
    print(f"   Tổng số ảnh tìm thấy: {stats['total_found']}")
    print(f"   Đã chuyển đổi: {stats['converted']}")
    print(f"   Đã đúng định dạng (bỏ qua): {stats['skipped']}")
    print(f"   Lỗi: {stats['errors']}")
    print(f"{'='*60}")
    
    return stats


if __name__ == "__main__":
    # Ví dụ sử dụng
    test_path = "/mnt/c/Users/Admin/HUIT - Học Tập/Năm 3/Semester_2/Research/Deeplearning/Yolo_Count/output_data/Apple/images"
    target_format = "jpg"
    
    print(f"🚀 Bắt đầu chuyển đổi ảnh trong thư mục: {test_path}")
    print(f"🎯 Định dạng đích: .{target_format}\n")
    
    result = format_image(test_path, target_format)
