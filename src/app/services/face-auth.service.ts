import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, timeout } from 'rxjs';

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceProfileSummary {
  profile_id: string;
  email: string;
  role: string;
  label: string;
  sample_count: number;
  created_at: string;
  updated_at: string;
}

export interface FaceProfilesResponse {
  profiles: FaceProfileSummary[];
}

export interface FaceRegistrationResponse {
  status: string;
  message: string;
  profile: FaceProfileSummary;
  sample_id: string;
}

export interface FaceRecognitionMatch {
  profile_id: string;
  email: string;
  role: string;
  label: string;
  sample_count: number;
  score: number;
  runner_up_margin: number;
}

export interface FaceRecognitionResponse {
  recognized: boolean;
  reason: string;
  message: string;
  confidence: number;
  face_box: FaceBox | null;
  match: FaceRecognitionMatch | null;
  profiles_enrolled: number;
}

@Injectable({ providedIn: 'root' })
export class FaceAuthService {
  private readonly listEndpoint = '/api/faces';
  private readonly registerEndpoint = '/api/faces/register';
  private readonly recognizeEndpoint = '/api/faces/recognize';

  constructor(private http: HttpClient) {}

  listProfiles(): Observable<FaceProfilesResponse> {
    return this.http.get<FaceProfilesResponse>(this.listEndpoint).pipe(timeout(10000));
  }

  registerFace(payload: { email: string; role: string; image: string; label?: string }): Observable<FaceRegistrationResponse> {
    return this.http.post<FaceRegistrationResponse>(this.registerEndpoint, payload).pipe(timeout(15000));
  }

  recognizeFace(payload: { image: string }): Observable<FaceRecognitionResponse> {
    return this.http.post<FaceRecognitionResponse>(this.recognizeEndpoint, payload).pipe(timeout(12000));
  }
}
