import glob
import itertools
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("OpenCV:", cv2.__version__)

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "images"
LEGACY_IMAGES_DIR = PROJECT_ROOT / "imagens"
CHECKERBOARD = (3, 3)  # cantos internos -- confirme contando no tabuleiro fisico
SQUARE_SIZE_MM = 90.0  # 9cm x 9cm, como voce informou


# Testa varias combinacoes de tamanhos de tabuleiro para encontrar o padrao correto.
def detect_board_size(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nao consegui abrir {os.path.basename(image_path)}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = list(itertools.product(range(3, 10), range(3, 10)))
    flags_classic = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    found_any = False
    for size in candidates:
        found, _ = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            print(f"SB detectou com CHECKERBOARD = {size}")
            found_any = True

        found, _ = cv2.findChessboardCorners(gray, size, flags_classic)
        if found:
            print(f"Classico detectou com CHECKERBOARD = {size}")
            found_any = True

    if not found_any:
        print(
            "Nenhuma combinacao (3..9 x 3..9) funcionou -- provavelmente nao e "
            "o CHECKERBOARD. Ver hipoteses abaixo."
        )

    return found_any


# Monta os pontos 3D do tabuleiro no sistema de coordenadas do mundo.
def build_object_points(checkerboard, square_size_mm):
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


# Realiza a calibracao da camera usando as imagens do tabuleiro.
def calibrate_single_camera(image_pattern, checkerboard, square_size_mm, show_visualization=False):
    image_paths = sorted(glob.glob(image_pattern))
    print(f"{len(image_paths)} imagens encontradas em {image_pattern}\n")

    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {image_pattern}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    object_points = build_object_points(checkerboard, square_size_mm)

    objpoints = []
    imgpoints = []
    used_paths = []
    img_shape = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Nao consegui abrir {os.path.basename(path)} -- caminho/arquivo invalido")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, checkerboard, flags)
        if not found:
            print(f"Tabuleiro NAO encontrado em {os.path.basename(path)} ({gray.shape[1]}x{gray.shape[0]}px) -- descartada")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(object_points)
        imgpoints.append(corners_refined)
        used_paths.append(path)

        if show_visualization:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, checkerboard, corners_refined, found)
            plt.figure(figsize=(5, 4))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(os.path.basename(path))
            plt.axis("off")
            plt.show()

    print(f"\n{len(used_paths)}/{len(image_paths)} imagens com deteccao bem-sucedida")

    if len(used_paths) == 0:
        print(
            "\nNenhuma imagem funcionou -- causas mais comuns, em ordem de "
            "probabilidade:\n"
            "  1. CHECKERBOARD errado: deve ser (cantos internos), nao "
            "(quadrados). Conte onde 4 quadrados se encontram, nao os "
            "quadrados em si.\n"
            "  2. Tabuleiro cortado na borda da foto (precisa aparecer "
            "inteiro).\n"
            "  3. Resolucao baixa demais / desfoque de movimento.\n"
            "  4. Tabuleiro muito inclinado/rotacionado em relacao a camera "
            "(tente fotos mais frontais)."
        )
        return None

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    print("Erro de reprojeccao medio (RMS) retornado pela calibracao:", ret)
    print("\nMatriz intrinseca da camera (K):")
    print(camera_matrix)
    print("\nCoeficientes de distorcao (k1, k2, p1, p2, k3):")
    print(dist_coeffs.ravel())

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"\nfx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "objpoints": objpoints,
        "imgpoints": imgpoints,
        "used_paths": used_paths,
    }


# Corrige a distorcao da imagem e exibe a comparacao antes/depois.
def undistort_sample(sample_path, camera_matrix, dist_coeffs):
    img = cv2.imread(sample_path)
    if img is None:
        raise FileNotFoundError(f"Nao consegui abrir {os.path.basename(sample_path)}")

    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, w_roi, h_roi = roi
    undistorted_cropped = undistorted[y:y + h_roi, x:x + w_roi]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (com distorcao)")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(undistorted_cropped, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Corrigida (sem distorcao)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


# Compara os pontos detectados com os pontos reprojetados para medir o erro de calibracao.
def evaluate_reprojection(calib_data):
    idx = 0
    sample_path = calib_data["used_paths"][idx]
    img = cv2.imread(sample_path)

    projected_points, _ = cv2.projectPoints(
        calib_data["objpoints"][idx],
        calib_data["rvecs"][idx],
        calib_data["tvecs"][idx],
        calib_data["camera_matrix"],
        calib_data["dist_coeffs"],
    )

    detected = calib_data["imgpoints"][idx].reshape(-1, 2)
    projected = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(detected - projected, axis=1)

    print(f"Imagem: {os.path.basename(sample_path)}")
    print(f"Erro de reprojeccao medio: {errors.mean():.3f} px  (maximo: {errors.max():.3f} px)")

    vis = img.copy()
    for (dx, dy), (px, py) in zip(detected, projected):
        cv2.circle(vis, (int(dx), int(dy)), 6, (0, 255, 0), 2)
        cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)

    plt.figure(figsize=(7, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Verde = ponto detectado na imagem | Vermelho = ponto 3D reprojetado")
    plt.axis("off")
    plt.show()

    print("\n| Ponto | Detectado (x,y) | Reprojetado (x,y) | Erro (px) |")
    print("|---|---|---|---|")
    for i, ((dx, dy), (px, py), err) in enumerate(zip(detected, projected, errors)):
        print(f"| {i} | ({dx:.1f}, {dy:.1f}) | ({px:.1f}, {py:.1f}) | {err:.3f} |")

    return errors


# Calibra um par de cameras estereoscopicas usando imagens do mesmo tabuleiro.
def calibrate_stereo(left_dir, right_dir, checkerboard, square_size_mm):
    left_paths = sorted(glob.glob(f"{left_dir}/*.jpg") + glob.glob(f"{left_dir}/*.png"))
    right_paths = sorted(glob.glob(f"{right_dir}/*.jpg") + glob.glob(f"{right_dir}/*.png"))

    if len(left_paths) != len(right_paths):
        raise ValueError("precisa do mesmo numero de imagens nas duas cameras, pareadas")

    objpoints_st = []
    imgpoints_left = []
    imgpoints_right = []
    object_points = build_object_points(checkerboard, square_size_mm)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    last_gray_left = None
    last_gray_right = None

    for lp, rp in zip(left_paths, right_paths):
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        last_gray_left = gray_l
        last_gray_right = gray_r

        found_l, corners_l = cv2.findChessboardCorners(gray_l, checkerboard, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, checkerboard, None)

        if not found_l or not found_r:
            continue

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        objpoints_st.append(object_points)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)

    print(f"{len(objpoints_st)} pares validos pra calibracao estereo")

    if not objpoints_st or last_gray_left is None or last_gray_right is None:
        raise ValueError("Nao foi possivel encontrar pares validos para calibracao estereo")

    _, k1, d1, _, _ = cv2.calibrateCamera(objpoints_st, imgpoints_left, last_gray_left.shape[::-1], None, None)
    _, k2, d2, _, _ = cv2.calibrateCamera(objpoints_st, imgpoints_right, last_gray_right.shape[::-1], None, None)

    flags = cv2.CALIB_FIX_INTRINSIC
    ret_st, k1, d1, k2, d2, r, t, e, f = cv2.stereoCalibrate(
        objpoints_st,
        imgpoints_left,
        imgpoints_right,
        k1,
        d1,
        k2,
        d2,
        last_gray_left.shape[::-1],
        criteria=criteria,
        flags=flags,
    )

    print("Erro RMS da calibracao estereo:", ret_st)
    print("\nR (rotacao camera direita em relacao a esquerda):\n", r)
    print("\nT (translacao, mesma unidade do SQUARE_SIZE_MM):\n", t.ravel())

    p1 = k1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    p2 = k2 @ np.hstack((r, t))
    point_left = imgpoints_left[-1][0].reshape(2, 1)
    point_right = imgpoints_right[-1][0].reshape(2, 1)
    point_4d = cv2.triangulatePoints(p1, p2, point_left, point_right)
    point_3d = (point_4d[:3] / point_4d[3]).ravel()

    print(f"\nPonto triangulado no espaco 3D (mesma unidade do SQUARE_SIZE_MM): {point_3d}")

    return {
        "ret_st": ret_st,
        "k1": k1,
        "d1": d1,
        "k2": k2,
        "d2": d2,
        "r": r,
        "t": t,
        "e": e,
        "f": f,
        "point_3d": point_3d,
    }


# Funcao principal que orquestra a calibracao monocular e estereo.
def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_patterns = [
        str(IMAGES_DIR / "*.jpg"),
        str(IMAGES_DIR / "*.png"),
        str(IMAGES_DIR / "*.jpeg"),
        str(LEGACY_IMAGES_DIR / "*.jpg"),
        str(LEGACY_IMAGES_DIR / "*.png"),
        str(LEGACY_IMAGES_DIR / "*.jpeg"),
    ]
    image_paths = sorted(
        [p for pattern in image_patterns for p in glob.glob(pattern)]
    )
    if image_paths:
        detect_board_size(image_paths[0])
    else:
        print(f"Nenhuma imagem encontrada em {IMAGES_DIR} ou {LEGACY_IMAGES_DIR}")
        return

    single_camera_pattern = next((pattern for pattern in image_patterns if glob.glob(pattern)), image_patterns[0])
    single_camera = calibrate_single_camera(single_camera_pattern, CHECKERBOARD, SQUARE_SIZE_MM, show_visualization=True)
    if single_camera is None:
        return

    if single_camera["used_paths"]:
        undistort_sample(single_camera["used_paths"][0], single_camera["camera_matrix"], single_camera["dist_coeffs"])
        evaluate_reprojection(single_camera)

    left_dir = PROJECT_ROOT / "calib_images_left"
    right_dir = PROJECT_ROOT / "calib_images_right"
    if left_dir.is_dir() and right_dir.is_dir():
        calibrate_stereo(str(left_dir), str(right_dir), CHECKERBOARD, SQUARE_SIZE_MM)
    else:
        print("Pastas estereo nao encontradas; pulando calibracao stereo.")


if __name__ == "__main__":
    main()
