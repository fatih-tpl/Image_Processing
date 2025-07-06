import cv2
import numpy as np
import matplotlib.pyplot as plt

# Göz referansı (mm cinsinden)
eye_diameter_mm = 24

# Resmi oku ve grayscale'e dönüştür
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

# Gaussian blur uygulama
def apply_gaussian_blur(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("Gaussian Filter", blurred)
    return blurred

# Corrosion and swelling (Erozyon ve dilasyon uygulama)
def erosion_dilation(blurred):
    kernel = np.ones((5, 5), np.uint8)                  # 5x5'lik birim kernel matrisi oluşturuldu
    eroded = cv2.erode(blurred, kernel, iterations=3)  # Görüntüdeki parlak alanları (beyaz pikselleri) aşındırır.
    cv2.imshow("Erosion Image", eroded)
    dilated = cv2.dilate(eroded, kernel, iterations=3)
    cv2.imshow("Dilation Image", dilated)
    return dilated

# Threshold segmentasyonu (Eşikleme)
def threshold_segmentation(dilated):
    _, thresholded = cv2.threshold(dilated, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold Image", thresholded)
    return thresholded

# Edge extraction (Kenarları çıkarma)
def edge_extraction(thresholded):
    edges = cv2.Canny(thresholded, 100, 200)             
    cv2.imshow("Edge Extraction Image", edges)
    return edges

# Elips fit etme ve çap ölçme
def fit_pupil_contour(img, edges, reference_eye_diameter_mm):
    # Kenarlardaki konturları bul
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # En büyük konturu seç
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Elips fit et
    if len(largest_contour) >= 5:                          # Elips fit etmek için en az 5 nokta gereklidir
        ellipse = cv2.fitEllipse(largest_contour)          # Elde edilen kontura elips oturtur
        center, (major_axis, minor_axis), angle = ellipse
        
        # Elipsin merkezi ve yarıçapları
        center = tuple(map(int, center))                   # Merkez (x, y)
        major_axis = int(major_axis)                       # Büyük eksen (x eksenindeki çap)
        minor_axis = int(minor_axis)                       # Küçük eksen (y eksenindeki çap)

        # Görüntüde elipsi çizme
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)          # Yeşil Renkte bir Elips çizme
        cv2.circle(img, center, 2, (0, 0, 255), 3)         # Elipsin merkezine bir kırmızı nokta
        cv2.putText(img, f'Major Axis: {major_axis}px', (center[0] - 50, center[1] - major_axis - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f'Minor Axis: {minor_axis}px', (center[0] - 50, center[1] - minor_axis + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"Major Axis (X ekseni çapı, piksel cinsinden): {major_axis}")
        print(f"Minor Axis (Y ekseni çapı, piksel cinsinden): {minor_axis}")
        
        # Referans Değeri ile Fiziksel çapı hesaplama
        eye_radius_pixels = 144                                      # Referans olarak gözün çapı piksel cinsinden karşılığı
        pixel_to_mm_ratio = eye_diameter_mm /( eye_radius_pixels)    # Piksel başına mm dönüşümü
        
        # Pupil çapını mm cinsinden hesapla
        pupil_diameter_mm_major = major_axis * pixel_to_mm_ratio
        pupil_diameter_mm_minor = minor_axis * pixel_to_mm_ratio
        print(f"X Axis (mm cinsinden): {pupil_diameter_mm_major:.2f} mm")
        print(f"Y Axis (mm cinsinden): {pupil_diameter_mm_minor:.2f} mm")
        
        return pupil_diameter_mm_major, pupil_diameter_mm_minor, img
    else:
        print("Elips fit etmek için yeterli sayıda nokta yok.")
        return None, None, img

# Ana fonksiyon
def main(image_path):
    img, gray_image = preprocess_image(image_path)
    
    blurred_image = apply_gaussian_blur(gray_image)
    dilated_image = erosion_dilation(blurred_image)
    thresholded_image = threshold_segmentation(dilated_image)
    edges = edge_extraction(thresholded_image)
    
    # Daireyi fit et ve çapı ölçme
    pupil_diameter_mm_major, pupil_diameter_mm_minor, result_img = fit_pupil_contour(img.copy(), edges, eye_diameter_mm)
    
    if pupil_diameter_mm_major and pupil_diameter_mm_minor:
        print(f"Major Axis (X ekseni çapi): {pupil_diameter_mm_major:.2f} mm")
        print(f"Minor Axis (Y ekseni çapi): {pupil_diameter_mm_minor:.2f} mm")
    
    cv2.imshow("Pupil Diameter Detection", result_img)
    
    plt.imshow(result_img)
    plt.title(f"X Axis: {pupil_diameter_mm_major:.2f} mm, Y Axis: {pupil_diameter_mm_minor:.2f} mm")
    plt.axis('off')
    plt.show()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "eye1(50).jpg"  # Resmin yolu

# Ana fonksiyonu çağırma
if __name__ == "__main__":
    main(image_path)
