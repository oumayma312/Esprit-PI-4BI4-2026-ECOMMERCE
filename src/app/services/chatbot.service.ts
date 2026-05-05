import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, TimeoutError, retry, timeout } from 'rxjs';

export interface ChatbotTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatbotResponse {
  answer: string;
  mode: string;
  model: string;
  sources: string[];
  response_ms: number;
}

export interface ApiHealthResponse {
  status: string;
  timestamp: string;
  chatbot: {
    ready: boolean;
    mode: string;
    model: string;
    data_dir: string;
    gemini_enabled: boolean;
    gemini_status: string;
  };
  face_recognition: {
    ready: boolean;
    profiles_enrolled: number;
    samples_enrolled: number;
  };
}

export interface ChatbotErrorDetails {
  kind: 'timeout' | 'offline' | 'server' | 'unknown';
  message: string;
  status?: number;
}

@Injectable({ providedIn: 'root' })
export class ChatbotService {
  private readonly endpoint = '/api/chatbot';
  private readonly healthEndpoint = '/api/health';

  constructor(private http: HttpClient) {}

  ask(message: string, history: ChatbotTurn[]): Observable<ChatbotResponse> {
    return this.http.post<ChatbotResponse>(this.endpoint, { message, history }).pipe(
      timeout(30000),
      retry({ count: 1, delay: 800 })
    );
  }

  checkHealth(): Observable<ApiHealthResponse> {
    return this.http.get<ApiHealthResponse>(this.healthEndpoint).pipe(timeout(5000));
  }

  describeError(error: unknown): ChatbotErrorDetails {
    if (error instanceof TimeoutError) {
      return {
        kind: 'timeout',
        message: 'The chatbot API took too long to respond. Please try again.'
      };
    }

    if (error instanceof HttpErrorResponse) {
      if (error.status === 0) {
        return {
          kind: 'offline',
          message: 'The chatbot API is unreachable. Run `npm start` or `npm run start:api` and try again.',
          status: 0
        };
      }

      const detail =
        typeof error.error?.detail === 'string'
          ? error.error.detail
          : 'The chatbot API returned an unexpected error.';

      return {
        kind: 'server',
        message: detail,
        status: error.status
      };
    }

    return {
      kind: 'unknown',
      message: 'The chatbot request could not be completed.'
    };
  }
}
