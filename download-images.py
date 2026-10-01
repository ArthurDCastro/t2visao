import os
import re
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGES_DIR = PROJECT_ROOT / "images"


def baixar_album_google_photos(link_compartilhado, pasta_destino=None):
    pasta_destino = Path(pasta_destino) if pasta_destino else IMAGES_DIR
    if not pasta_destino.is_absolute():
        pasta_destino = PROJECT_ROOT / pasta_destino

    pasta_destino.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(link_compartilhado, headers=headers, allow_redirects=True)
    resp.raise_for_status()

    urls = re.findall(r'"(https://lh3\.googleusercontent\.com/[^"]+?)"', resp.text)
    urls = list(dict.fromkeys(urls))

    if not urls:
        raise RuntimeError("Não foi encontrada nenhuma imagem na página -- confira se o link é público.")

    print(f"{len(urls)} imagens encontradas no álbum")

    caminhos = []
    for i, url in enumerate(urls, start=1):
        url_full = url.split("=")[0] + "=d"
        img_resp = requests.get(url_full, headers=headers)
        img_resp.raise_for_status()

        caminho = pasta_destino / f"foto{i}.jpg"
        with open(caminho, "wb") as f:
            f.write(img_resp.content)
        print(f"Imagem salva em: {caminho}")
        caminhos.append(str(caminho))

    return caminhos


if __name__ == "__main__":
    baixar_album_google_photos("https://photos.app.goo.gl/7z7o1y6Q2E372HZj6", pasta_destino=IMAGES_DIR)
