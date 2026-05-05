from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FaceServiceError(Exception):
    pass


class FaceValidationError(FaceServiceError):
    pass


class FaceNotFoundError(FaceServiceError):
    pass


class MultipleFacesDetectedError(FaceServiceError):
    pass


@dataclass(frozen=True)
class DetectedFace:
    box: dict[str, int]
    face_gray: np.ndarray
    quality: dict[str, float]
    quality_ok: bool
    quality_message: str | None


class FaceRegistry:
    MIN_RECOGNITION_SCORE = 0.82
    MIN_RUNNER_UP_MARGIN = 0.08

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.samples_dir = self.storage_dir / "samples"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("The OpenCV face cascade could not be loaded.")

        self.hog = cv2.HOGDescriptor()

    def health_summary(self) -> dict[str, Any]:
        profiles = self.list_profiles()
        return {
            "ready": True,
            "profiles_enrolled": len(profiles),
            "samples_enrolled": int(sum(int(profile["sample_count"]) for profile in profiles)),
        }

    def list_profiles(self) -> list[dict[str, Any]]:
        return [self._profile_summary(profile) for profile in self._load_profiles(migrate_legacy=True)]

    def register(self, email: str, role: str, image_data_url: str, label: str | None = None) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        normalized_role = role.strip().lower()
        if not normalized_email:
            raise FaceValidationError("An email is required to save a face profile.")
        if not normalized_role:
            raise FaceValidationError("A role is required to save a face profile.")

        detected_face = self._extract_face(self._decode_image(image_data_url))
        if not detected_face.quality_ok:
            raise FaceValidationError(detected_face.quality_message or "The captured face is not clear enough.")

        sample_id = uuid4().hex
        sample_path = self.samples_dir / f"{sample_id}.png"
        cv2.imwrite(str(sample_path), detected_face.face_gray)

        features = self._encode_face(detected_face.face_gray)
        sample = {
            "sample_id": sample_id,
            "created_at": utc_now_iso(),
            "face_box": detected_face.box,
            "image_path": str(Path("samples") / sample_path.name).replace("\\", "/"),
            "quality": detected_face.quality,
            "features": features,
        }

        profile, profile_path = self._find_profile_by_email(normalized_email)
        now = utc_now_iso()
        if profile is None or profile_path is None:
            profile_id = uuid4().hex
            profile = {
                "profile_id": profile_id,
                "email": normalized_email,
                "role": normalized_role,
                "label": label.strip() if label and label.strip() else normalized_role.upper(),
                "created_at": now,
                "updated_at": now,
                "samples": [sample],
            }
            profile_path = self.storage_dir / f"face_profile_{profile_id}.json"
        else:
            profile["role"] = normalized_role
            if label and label.strip():
                profile["label"] = label.strip()
            profile["updated_at"] = now
            profile.setdefault("samples", []).append(sample)

        self._save_profile(profile, profile_path)
        return {
            "status": "ok",
            "message": f"Face sample saved for {normalized_email}.",
            "profile": self._profile_summary(profile),
            "sample_id": sample_id,
        }

    def recognize(self, image_data_url: str) -> dict[str, Any]:
        profiles = self._load_profiles(migrate_legacy=True)
        if not profiles:
            return {
                "recognized": False,
                "reason": "no_profiles",
                "message": "No face profiles are enrolled yet.",
                "confidence": 0.0,
                "face_box": None,
                "match": None,
                "profiles_enrolled": 0,
            }

        try:
            detected_face = self._extract_face(self._decode_image(image_data_url))
        except FaceNotFoundError:
            return {
                "recognized": False,
                "reason": "no_face",
                "message": "No face was detected in the current frame.",
                "confidence": 0.0,
                "face_box": None,
                "match": None,
                "profiles_enrolled": len(profiles),
            }
        except MultipleFacesDetectedError:
            return {
                "recognized": False,
                "reason": "multiple_faces",
                "message": "Multiple faces were detected. Only one face is allowed for authentication.",
                "confidence": 0.0,
                "face_box": None,
                "match": None,
                "profiles_enrolled": len(profiles),
            }

        if not detected_face.quality_ok:
            return {
                "recognized": False,
                "reason": "low_quality",
                "message": detected_face.quality_message or "The detected face is too blurry or too dark.",
                "confidence": 0.0,
                "face_box": detected_face.box,
                "match": None,
                "profiles_enrolled": len(profiles),
            }

        probe_features = self._encode_face(detected_face.face_gray)
        scored_profiles: list[dict[str, Any]] = []
        for profile in profiles:
            sample_scores: list[float] = []
            for sample in profile.get("samples", []):
                features = sample.get("features")
                if not isinstance(features, dict):
                    continue
                sample_scores.append(self._compare_features(probe_features, features))
            if not sample_scores:
                continue
            sample_scores.sort(reverse=True)
            top_scores = sample_scores[: min(3, len(sample_scores))]
            profile_score = float(sum(top_scores) / len(top_scores))
            scored_profiles.append(
                {
                    "profile": profile,
                    "score": profile_score,
                    "sample_count": len(profile.get("samples", [])),
                }
            )

        if not scored_profiles:
            return {
                "recognized": False,
                "reason": "no_match_data",
                "message": "No usable enrolled face samples were found on the server.",
                "confidence": 0.0,
                "face_box": detected_face.box,
                "match": None,
                "profiles_enrolled": len(profiles),
            }

        scored_profiles.sort(key=lambda item: item["score"], reverse=True)
        best = scored_profiles[0]
        runner_up_score = float(scored_profiles[1]["score"]) if len(scored_profiles) > 1 else 0.0
        margin = float(best["score"] - runner_up_score)
        recognized = bool(
            best["score"] >= self.MIN_RECOGNITION_SCORE and margin >= self.MIN_RUNNER_UP_MARGIN
        )

        best_profile = best["profile"]
        match = {
            "profile_id": best_profile["profile_id"],
            "email": best_profile["email"],
            "role": best_profile["role"],
            "label": best_profile.get("label") or best_profile["role"],
            "sample_count": best["sample_count"],
            "score": round(float(best["score"]), 4),
            "runner_up_margin": round(margin, 4),
        }

        return {
            "recognized": recognized,
            "reason": "recognized" if recognized else "unknown_face",
            "message": (
                f"Recognized: {match['label']}."
                if recognized
                else "The detected face does not match any enrolled profile with enough confidence."
            ),
            "confidence": round(float(best["score"]), 4),
            "face_box": detected_face.box,
            "match": match,
            "profiles_enrolled": len(profiles),
        }

    def _load_profiles(self, migrate_legacy: bool) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for path in sorted(self.storage_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            normalized = self._normalize_profile(raw, path, migrate_legacy=migrate_legacy)
            if normalized is not None:
                profiles.append(normalized)
        return profiles

    def _find_profile_by_email(self, email: str) -> tuple[dict[str, Any] | None, Path | None]:
        for path in sorted(self.storage_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            normalized = self._normalize_profile(raw, path, migrate_legacy=True)
            if normalized is not None and normalized.get("email") == email:
                return normalized, path
        return None, None

    def _normalize_profile(
        self, raw: dict[str, Any], path: Path, migrate_legacy: bool
    ) -> dict[str, Any] | None:
        if raw.get("samples"):
            normalized_samples: list[dict[str, Any]] = []
            has_changed = False
            for sample in raw.get("samples", []):
                normalized_sample = self._normalize_sample(sample)
                if normalized_sample is None:
                    continue
                normalized_samples.append(normalized_sample)
                if normalized_sample != sample:
                    has_changed = True

            if not normalized_samples:
                return None

            profile = {
                "profile_id": str(raw.get("profile_id") or path.stem),
                "email": str(raw.get("email", "")).strip().lower(),
                "role": str(raw.get("role", "unknown")).strip().lower() or "unknown",
                "label": str(raw.get("label") or raw.get("role") or "unknown").strip(),
                "created_at": str(raw.get("created_at") or raw.get("registered_at") or utc_now_iso()),
                "updated_at": str(raw.get("updated_at") or utc_now_iso()),
                "samples": normalized_samples,
            }
            if migrate_legacy and has_changed:
                self._save_profile(profile, path)
            return profile

        legacy_image = raw.get("image")
        if not isinstance(legacy_image, str) or not legacy_image.strip():
            return None

        try:
            detected_face = self._extract_face(self._decode_image(legacy_image))
        except FaceServiceError:
            return None

        sample_id = str(raw.get("sample_id") or uuid4().hex)
        sample_path = self.samples_dir / f"{sample_id}.png"
        cv2.imwrite(str(sample_path), detected_face.face_gray)
        profile_id = str(raw.get("profile_id") or uuid4().hex)
        profile = {
            "profile_id": profile_id,
            "email": str(raw.get("email", "")).strip().lower(),
            "role": str(raw.get("role", "unknown")).strip().lower() or "unknown",
            "label": str(raw.get("label") or raw.get("role") or "unknown").strip(),
            "created_at": str(raw.get("created_at") or raw.get("registered_at") or utc_now_iso()),
            "updated_at": utc_now_iso(),
            "samples": [
                {
                    "sample_id": sample_id,
                    "created_at": str(raw.get("registered_at") or utc_now_iso()),
                    "face_box": detected_face.box,
                    "image_path": str(Path("samples") / sample_path.name).replace("\\", "/"),
                    "quality": detected_face.quality,
                    "features": self._encode_face(detected_face.face_gray),
                }
            ],
        }
        if migrate_legacy:
            self._save_profile(profile, path)
        return profile

    def _normalize_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        sample_id = str(sample.get("sample_id") or uuid4().hex)
        image_path = sample.get("image_path")
        resolved_image_path = self._resolve_sample_image_path(image_path) if isinstance(image_path, str) else None

        features = sample.get("features")
        if not isinstance(features, dict) and resolved_image_path is not None:
            face_gray = cv2.imread(str(resolved_image_path), cv2.IMREAD_GRAYSCALE)
            if face_gray is not None:
                features = self._encode_face(face_gray)

        if not isinstance(features, dict):
            return None

        normalized = {
            "sample_id": sample_id,
            "created_at": str(sample.get("created_at") or utc_now_iso()),
            "face_box": self._normalize_box(sample.get("face_box")),
            "image_path": (
                str(Path("samples") / resolved_image_path.name).replace("\\", "/")
                if resolved_image_path is not None
                else str(image_path or "")
            ),
            "quality": self._normalize_quality(sample.get("quality")),
            "features": self._normalize_features(features),
        }
        return normalized

    def _save_profile(self, profile: dict[str, Any], path: Path) -> None:
        serializable = {
            **profile,
            "samples": [
                {
                    **sample,
                    "quality": self._normalize_quality(sample.get("quality")),
                    "features": self._normalize_features(sample.get("features")),
                }
                for sample in profile.get("samples", [])
            ],
        }
        path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "profile_id": profile["profile_id"],
            "email": profile["email"],
            "role": profile["role"],
            "label": profile.get("label") or profile["role"],
            "sample_count": len(profile.get("samples", [])),
            "created_at": profile.get("created_at"),
            "updated_at": profile.get("updated_at"),
        }

    def _decode_image(self, image_data_url: str) -> np.ndarray:
        payload = image_data_url.strip()
        if not payload:
            raise FaceValidationError("The captured frame is empty.")

        if "," in payload:
            payload = payload.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(payload)
        except Exception as exc:  # noqa: BLE001
            raise FaceValidationError("The captured frame could not be decoded.") from exc

        image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise FaceValidationError("The captured frame is not a valid image.")
        return image

    def _extract_face(self, image: np.ndarray) -> DetectedFace:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=7,
            minSize=(90, 90),
        )

        if len(faces) == 0:
            raise FaceNotFoundError("No face was detected.")
        if len(faces) > 1:
            raise MultipleFacesDetectedError("More than one face was detected.")

        x, y, width, height = (int(value) for value in faces[0])
        padding_x = int(width * 0.18)
        padding_y = int(height * 0.22)
        x0 = max(0, x - padding_x)
        y0 = max(0, y - padding_y)
        x1 = min(gray.shape[1], x + width + padding_x)
        y1 = min(gray.shape[0], y + height + padding_y)
        face_gray = gray[y0:y1, x0:x1]
        if face_gray.size == 0:
            raise FaceNotFoundError("The detected face could not be cropped.")

        normalized_face = cv2.resize(face_gray, (160, 160), interpolation=cv2.INTER_AREA)
        quality = self._measure_quality(normalized_face)
        quality_ok, quality_message = self._quality_gate(quality)
        return DetectedFace(
            box={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            face_gray=normalized_face,
            quality=quality,
            quality_ok=quality_ok,
            quality_message=quality_message,
        )

    @staticmethod
    def _measure_quality(face_gray: np.ndarray) -> dict[str, float]:
        sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        brightness = float(face_gray.mean())
        contrast = float(face_gray.std())
        return {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
        }

    @staticmethod
    def _quality_gate(quality: dict[str, float]) -> tuple[bool, str | None]:
        sharpness = float(quality.get("sharpness", 0.0))
        brightness = float(quality.get("brightness", 0.0))
        contrast = float(quality.get("contrast", 0.0))

        if sharpness < 35:
            return False, "Face detected, but the image is too blurry."
        if brightness < 45:
            return False, "Face detected, but the scene is too dark."
        if brightness > 215:
            return False, "Face detected, but the scene is too bright."
        if contrast < 22:
            return False, "Face detected, but the contrast is too low."
        return True, None

    def _encode_face(self, face_gray: np.ndarray) -> dict[str, list[float]]:
        normalized_face = cv2.equalizeHist(face_gray)
        hog_input = cv2.resize(normalized_face, (64, 128), interpolation=cv2.INTER_AREA)
        hog_features = self.hog.compute(hog_input)
        if hog_features is None:
            raise FaceValidationError("The face features could not be computed.")

        histogram = cv2.calcHist([normalized_face], [0], None, [32], [0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        thumbnail = cv2.resize(normalized_face, (32, 32), interpolation=cv2.INTER_AREA)
        thumbnail = thumbnail.astype(np.float32).flatten() / 255.0

        return {
            "hog": self._round_array(hog_features.flatten()),
            "histogram": self._round_array(histogram),
            "thumbnail": self._round_array(thumbnail),
        }

    def _compare_features(self, probe: dict[str, Any], reference: dict[str, Any]) -> float:
        probe_hog = np.asarray(probe["hog"], dtype=np.float32)
        reference_hog = np.asarray(reference["hog"], dtype=np.float32)
        probe_hist = np.asarray(probe["histogram"], dtype=np.float32)
        reference_hist = np.asarray(reference["histogram"], dtype=np.float32)
        probe_thumb = np.asarray(probe["thumbnail"], dtype=np.float32)
        reference_thumb = np.asarray(reference["thumbnail"], dtype=np.float32)

        hog_similarity = self._cosine_similarity(probe_hog, reference_hog)
        histogram_similarity = float(
            cv2.compareHist(probe_hist, reference_hist, cv2.HISTCMP_CORREL)
        )
        histogram_similarity = max(0.0, min(1.0, (histogram_similarity + 1.0) / 2.0))
        thumbnail_similarity = max(0.0, min(1.0, 1.0 - float(np.mean(np.abs(probe_thumb - reference_thumb)))))

        score = 0.62 * hog_similarity + 0.2 * histogram_similarity + 0.18 * thumbnail_similarity
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator == 0:
            return 0.0
        similarity = float(np.dot(first, second) / denominator)
        return max(0.0, min(1.0, similarity))

    def _resolve_sample_image_path(self, image_path: str) -> Path | None:
        candidate = (self.storage_dir / image_path).resolve()
        if candidate.exists():
            return candidate
        return None

    @staticmethod
    def _round_array(values: np.ndarray, digits: int = 6) -> list[float]:
        return [round(float(value), digits) for value in values.tolist()]

    @staticmethod
    def _normalize_box(raw_box: Any) -> dict[str, int]:
        if not isinstance(raw_box, dict):
            return {"x": 0, "y": 0, "width": 0, "height": 0}
        return {
            "x": int(raw_box.get("x", 0)),
            "y": int(raw_box.get("y", 0)),
            "width": int(raw_box.get("width", 0)),
            "height": int(raw_box.get("height", 0)),
        }

    @staticmethod
    def _normalize_quality(raw_quality: Any) -> dict[str, float]:
        if not isinstance(raw_quality, dict):
            return {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0}
        return {
            "sharpness": round(float(raw_quality.get("sharpness", 0.0)), 2),
            "brightness": round(float(raw_quality.get("brightness", 0.0)), 2),
            "contrast": round(float(raw_quality.get("contrast", 0.0)), 2),
        }

    def _normalize_features(self, raw_features: Any) -> dict[str, list[float]]:
        if not isinstance(raw_features, dict):
            raise FaceValidationError("The stored face features are invalid.")

        def normalize_vector(key: str) -> list[float]:
            values = raw_features.get(key, [])
            if not isinstance(values, list):
                raise FaceValidationError(f"The stored '{key}' vector is invalid.")
            return [round(float(value), 6) for value in values]

        return {
            "hog": normalize_vector("hog"),
            "histogram": normalize_vector("histogram"),
            "thumbnail": normalize_vector("thumbnail"),
        }
