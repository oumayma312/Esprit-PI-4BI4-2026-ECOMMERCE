import { HttpErrorResponse } from '@angular/common/http';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService, UserAccount } from '../services/auth.service';
import {
  FaceAuthService,
  FaceBox,
  FaceProfileSummary,
  FaceRecognitionResponse
} from '../services/face-auth.service';

interface OverlayBoxStyle {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface BrowserFaceDetection {
  boundingBox: DOMRectReadOnly;
}

interface BrowserFaceDetector {
  detect(input: ImageBitmapSource): Promise<BrowserFaceDetection[]>;
}

declare global {
  interface Window {
    FaceDetector?: new (options?: { fastMode?: boolean; maxDetectedFaces?: number }) => BrowserFaceDetector;
  }
}

type MessageTone = 'info' | 'success' | 'warning' | 'error';

@Component({
  selector: 'app-face-id',
  templateUrl: './face-id.component.html',
  styleUrls: ['./face-id.component.css']
})
export class FaceIdComponent implements OnInit, OnDestroy {
  @ViewChild('video') videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('stage') stageRef!: ElementRef<HTMLDivElement>;

  capturing = false;
  enrollInProgress = false;
  message = 'Start the camera to scan or enroll a face.';
  messageTone: MessageTone = 'info';
  profiles: FaceProfileSummary[] = [];
  accounts: UserAccount[] = [];
  selectedEmail = '';
  overlayBox: OverlayBoxStyle | null = null;

  private detector: BrowserFaceDetector | null = null;
  private stream: MediaStream | null = null;
  private overlayLoopId: number | null = null;
  private overlayBusy = false;
  private lastOverlayCheckAt = 0;
  private recognitionIntervalId: number | null = null;
  private recognitionBusy = false;
  private confirmationTrail: string[] = [];

  constructor(
    private auth: AuthService,
    private router: Router,
    private faceAuth: FaceAuthService
  ) {}

  ngOnInit(): void {
    this.accounts = this.auth.listAccounts();
    this.selectedEmail = this.accounts[0]?.email ?? '';
    this.detector = this.createBrowserDetector();
    this.loadProfiles();
  }

  ngOnDestroy(): void {
    this.stop();
  }

  async start(): Promise<void> {
    if (this.capturing) {
      return;
    }

    const video = this.videoRef.nativeElement;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });
      video.srcObject = this.stream;
      await video.play();
      this.capturing = true;
      this.confirmationTrail = [];
      this.setMessage(
        this.profiles.length
          ? 'Analyse en cours. Regardez la camera pour verifier votre identite.'
          : 'Camera active. Enregistrez d abord au moins un profil.',
        'info'
      );
      this.startOverlayLoop();
      this.startRecognitionLoop();
    } catch {
      this.setMessage('Camera access is unavailable on this device.', 'error');
      this.stop();
    }
  }

  stop(): void {
    this.capturing = false;
    this.confirmationTrail = [];
    this.overlayBox = null;

    if (this.overlayLoopId !== null) {
      window.cancelAnimationFrame(this.overlayLoopId);
      this.overlayLoopId = null;
    }

    if (this.recognitionIntervalId !== null) {
      window.clearInterval(this.recognitionIntervalId);
      this.recognitionIntervalId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    if (this.videoRef?.nativeElement) {
      this.videoRef.nativeElement.srcObject = null;
    }
  }

  enroll(email: string): void {
    if (!this.capturing) {
      this.setMessage('Start the camera before saving a face profile.', 'warning');
      return;
    }

    const account = this.auth.findByEmail(email);
    if (!account) {
      this.setMessage('Choose a valid user account before enrollment.', 'warning');
      return;
    }

    const image = this.captureCurrentFrame();
    if (!image) {
      this.setMessage('A frame could not be captured from the camera.', 'error');
      return;
    }

    this.enrollInProgress = true;
    this.setMessage(`Saving a face sample for ${account.role.toUpperCase()}...`, 'info');

    this.faceAuth
      .registerFace({
        email: account.email,
        role: account.role,
        label: account.role.toUpperCase(),
        image
      })
      .subscribe({
        next: response => {
          this.enrollInProgress = false;
          this.setMessage(
            `Profil enregistre pour ${response.profile.label}. ${response.profile.sample_count} capture(s) disponible(s).`,
            'success'
          );
          this.loadProfiles();
        },
        error: error => {
          this.enrollInProgress = false;
          this.setMessage(this.describeApiError(error, 'The face profile could not be saved.'), 'error');
        }
      });
  }

  trackProfile(index: number, profile: FaceProfileSummary): string {
    return profile.profile_id;
  }

  private loadProfiles(): void {
    this.faceAuth.listProfiles().subscribe({
      next: response => {
        this.profiles = [...response.profiles].sort((first, second) =>
          first.label.localeCompare(second.label)
        );
      },
      error: error => {
        this.profiles = [];
        this.setMessage(
          this.describeApiError(error, 'The enrolled face profiles could not be loaded.'),
          'error'
        );
      }
    });
  }

  private startRecognitionLoop(): void {
    if (this.recognitionIntervalId !== null) {
      return;
    }

    this.recognitionIntervalId = window.setInterval(() => {
      void this.recognizeCurrentFrame();
    }, 1100);
  }

  private async recognizeCurrentFrame(): Promise<void> {
    if (!this.capturing || this.recognitionBusy) {
      return;
    }

    const image = this.captureCurrentFrame();
    if (!image) {
      return;
    }

    this.recognitionBusy = true;
    this.faceAuth.recognizeFace({ image }).subscribe({
      next: response => {
        this.recognitionBusy = false;
        this.handleRecognitionResponse(response);
      },
      error: error => {
        this.recognitionBusy = false;
        this.confirmationTrail = [];
        this.setMessage(
          this.describeApiError(error, 'The face recognition service is unavailable.'),
          'error'
        );
      }
    });
  }

  private handleRecognitionResponse(response: FaceRecognitionResponse): void {
    if (response.face_box) {
      this.overlayBox = this.mapSourceBoxToStage(response.face_box);
    } else if (!this.detector) {
      this.overlayBox = null;
    }

    if (response.recognized && response.match) {
      const confirmed = this.pushConfirmation(response.match.email);
      if (confirmed) {
        this.setMessage(`Visage reconnu: ${response.match.label}. Redirection en cours...`, 'success');
        this.loginByEmail(response.match.email);
        return;
      }

      this.setMessage(
        `Visage reconnu: ${response.match.label}. Confirmation sur plusieurs frames...`,
        'success'
      );
      return;
    }

    this.confirmationTrail = [];

    switch (response.reason) {
      case 'no_profiles':
        this.setMessage('Aucun profil visage n est enregistre pour le moment.', 'warning');
        break;
      case 'no_face':
        this.setMessage('Aucun visage clair detecte.', 'info');
        if (!this.detector) {
          this.overlayBox = null;
        }
        break;
      case 'multiple_faces':
        this.setMessage('Plusieurs visages detectes. Une seule personne doit etre dans le cadre.', 'warning');
        break;
      case 'low_quality':
        this.setMessage('Visage detecte, mais la qualite de l image est insuffisante.', 'warning');
        break;
      case 'unknown_face':
        this.setMessage('Visage inconnu. Aucun profil correspondant avec un niveau de confiance suffisant.', 'warning');
        break;
      default:
        this.setMessage(response.message || 'Face recognition is running.', 'info');
        break;
    }
  }

  private pushConfirmation(email: string): boolean {
    this.confirmationTrail = [...this.confirmationTrail, email].slice(-3);
    return this.confirmationTrail.length === 3 && this.confirmationTrail.every(item => item === email);
  }

  private captureCurrentFrame(): string | null {
    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    const context = canvas.getContext('2d');
    if (!context || !video.videoWidth || !video.videoHeight) {
      return null;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.92);
  }

  private startOverlayLoop(): void {
    if (this.overlayLoopId !== null) {
      return;
    }

    const tick = async () => {
      if (!this.capturing) {
        return;
      }

      const video = this.videoRef.nativeElement;
      if (
        this.detector &&
        !this.overlayBusy &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        performance.now() - this.lastOverlayCheckAt > 150
      ) {
        this.overlayBusy = true;
        this.lastOverlayCheckAt = performance.now();
        try {
          const faces = await this.detector.detect(video);
          if (faces.length > 0) {
            const firstFace = faces[0];
            this.overlayBox = this.mapSourceBoxToStage({
              x: Math.round(firstFace.boundingBox.x),
              y: Math.round(firstFace.boundingBox.y),
              width: Math.round(firstFace.boundingBox.width),
              height: Math.round(firstFace.boundingBox.height)
            });
          } else if (!this.recognitionBusy) {
            this.overlayBox = null;
          }
        } catch {
          this.detector = null;
        } finally {
          this.overlayBusy = false;
        }
      }

      this.overlayLoopId = window.requestAnimationFrame(() => {
        void tick();
      });
    };

    void tick();
  }

  private createBrowserDetector(): BrowserFaceDetector | null {
    if (typeof window.FaceDetector !== 'function') {
      return null;
    }

    try {
      return new window.FaceDetector({
        fastMode: true,
        maxDetectedFaces: 1
      });
    } catch {
      return null;
    }
  }

  private mapSourceBoxToStage(sourceBox: FaceBox): OverlayBoxStyle | null {
    const stage = this.stageRef?.nativeElement;
    const video = this.videoRef?.nativeElement;
    if (!stage || !video || !video.videoWidth || !video.videoHeight) {
      return null;
    }

    const stageWidth = stage.clientWidth;
    const stageHeight = stage.clientHeight;
    if (!stageWidth || !stageHeight) {
      return null;
    }

    const scale = Math.min(stageWidth / video.videoWidth, stageHeight / video.videoHeight);
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    const offsetLeft = (stageWidth - renderedWidth) / 2;
    const offsetTop = (stageHeight - renderedHeight) / 2;

    return {
      left: offsetLeft + sourceBox.x * scale,
      top: offsetTop + sourceBox.y * scale,
      width: sourceBox.width * scale,
      height: sourceBox.height * scale
    };
  }

  private describeApiError(error: unknown, fallback: string): string {
    if (error instanceof HttpErrorResponse) {
      if (typeof error.error?.detail === 'string') {
        return error.error.detail;
      }

      if (error.status === 0) {
        return 'The local API is unreachable. Start it with `npm start` or `npm run start:api`.';
      }
    }

    return fallback;
  }

  private setMessage(message: string, tone: MessageTone): void {
    this.message = message;
    this.messageTone = tone;
  }

  private loginByEmail(email: string): void {
    const account = this.auth.findByEmail(email);
    if (!account) {
      this.setMessage('The recognized account is not linked to a local user.', 'error');
      return;
    }

    this.stop();
    localStorage.setItem('pi_user', JSON.stringify(account));
    this.router.navigate(['/dashboard', account.role]);
  }
}
