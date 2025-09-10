import cv2
import numpy as np

def main(video_path=0):
    # 0 = webcam, o pasá "video.mp4"
    cap = cv2.VideoCapture('test1.mp4')
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_kp, prev_des = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if prev_des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            matched_img = cv2.drawMatches(prev_frame, prev_kp, frame, kp, matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matches", matched_img)

        prev_kp, prev_des, prev_frame = kp, des, frame.copy()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
