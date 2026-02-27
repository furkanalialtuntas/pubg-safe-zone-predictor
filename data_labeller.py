import cv2
import math
import csv
import os
import glob

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(param['img_display'], (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(param['window_name'], param['img_display'])


def get_clicks(img_path, window_name, num_clicks, instruction, text_color=(0, 255, 0)):
    global clicked_points
    clicked_points = []

    img = cv2.imread(img_path)
    if img is None:
        print(f"Hata: {img_path} bulunamadı!")
        return None

    img_display = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(instruction, font, 1.2, 3)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = 50

    cv2.rectangle(img_display, (text_x - 10, text_y - 40), (text_x + text_size[0] + 10, text_y + 15), (0, 0, 0), -1)
    cv2.putText(img_display, instruction, (text_x, text_y), font, 1.2, text_color, 3)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_display)
    print(f"\n---> {instruction} <---")
    cv2.setMouseCallback(window_name, mouse_callback, {'img_display': img_display, 'window_name': window_name})

    while len(clicked_points) < num_clicks:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyWindow(window_name)
            return None

    cv2.destroyWindow(window_name)
    return clicked_points.copy()


def main():
    csv_filename   = "zone_data.csv"
    master_map_path = os.path.join("reference", "erangel_reference.png")

    if not os.path.exists(master_map_path):
        print(f"Hata: Master Map bulunamadı! '{master_map_path}' yolunu kontrol et.")
        return

    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Match_ID", "Phase", "White_X", "White_Y", "White_R"])

    image_files = sorted(glob.glob(os.path.join("raw", "mac*.png")))

    if not image_files:
        print("Hata: 'raw' klasöründe 'mac' ile başlayan .png dosyası yok!")
        return

    print(f"Toplam {len(image_files)} adet görsel bulundu. İşlem başlıyor...\n")

    for match_img_path in image_files:
        filename = os.path.basename(match_img_path)
        print(f"\n{'='*50}\nİŞLENEN GÖRSEL: {filename}\n{'='*50}")

        try:
            name_parts = filename.replace(".png", "").split("_")
            match_id   = name_parts[0]
            phase      = name_parts[1].replace("faz", "")
        except Exception:
            match_id = input("Match ID girin: ")
            phase    = input("Faz girin: ")

        screen_pts = get_clicks(match_img_path, f"Mac: {filename}", 2,
                                "ADIM 1: Haritada 2 UZAK referans noktasina tikla", (0, 255, 255))
        if not screen_pts:
            break

        master_pts = get_clicks(master_map_path, "Master Map", 2,
                                "ADIM 2: Master Map'te AYNI 2 noktaya tikla", (0, 165, 255))
        if not master_pts:
            break

        d_screen = math.hypot(screen_pts[1][0] - screen_pts[0][0], screen_pts[1][1] - screen_pts[0][1])
        d_master = math.hypot(master_pts[1][0] - master_pts[0][0], master_pts[1][1] - master_pts[0][1])

        if d_screen == 0:
            continue

        scale = d_master / d_screen
        t_x   = master_pts[0][0] - (screen_pts[0][0] * scale)
        t_y   = master_pts[0][1] - (screen_pts[0][1] * scale)

        zone_pts = get_clicks(match_img_path, "Alan Etiketleme", 2,
                              "ADIM 3: Beyaz cemberin once MERKEZINE, sonra CIZGISINE tikla", (50, 205, 50))
        if not zone_pts:
            break

        screen_center = zone_pts[0]
        screen_edge   = zone_pts[1]
        screen_r = math.hypot(screen_edge[0] - screen_center[0], screen_edge[1] - screen_center[1])

        real_x = int(screen_center[0] * scale + t_x)
        real_y = int(screen_center[1] * scale + t_y)
        real_r = int(screen_r * scale)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([match_id, phase, real_x, real_y, real_r])

        print(f"--> {filename} OK! Normalize edilmis veri kaydedildi.\n")

    print("\nİşlem tamamlandı!")


if __name__ == "__main__":
    main()