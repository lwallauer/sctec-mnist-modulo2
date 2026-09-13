"""
Pré-processamento de imagens autorais para o formato MNIST (28×28, [0, 1]).

Pipeline alinhado às diretrizes do mini-projeto:
  1. Leitura em escala de cinza
  2. Binarização (Otsu) + inversão (dígito branco em fundo preto)
  3. Bounding box do traço
  4. Redimensionamento proporcional (maior lado ≈ 20 px)
  5. Centralização em canvas 28×28
  6. Normalização para [0.0, 1.0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

PathLike = Union[str, Path]


def preprocess_image_to_mnist(
    image: PathLike | np.ndarray,
    *,
    target_size: int = 28,
    digit_span: float = 20.0,
) -> np.ndarray | None:
    """
    Converte imagem (caminho ou matriz) para matriz 28×28 float32 em [0, 1].

    Returns
    -------
    np.ndarray | None
        Matriz (28, 28) normalizada, ou None se a imagem for inválida/vazia.
    """
    if isinstance(image, (str, Path)):
        gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    else:
        gray = np.asarray(image)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.uint8)

    if gray is None or gray.size == 0:
        return None

    # Otsu + inversão: traço claro sobre fundo escuro (padrão MNIST)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y : y + h, x : x + w]

    scale = digit_span / max(w, h, 1)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    return canvas.astype(np.float32) / 255.0


def image_to_feature_vector(
    image: PathLike | np.ndarray,
    **kwargs,
) -> np.ndarray | None:
    """
    Retorna vetor (1, 784) pronto para inferência, ou None se inválido.
    """
    processed = preprocess_image_to_mnist(image, **kwargs)
    if processed is None:
        return None
    return processed.reshape(1, -1)
