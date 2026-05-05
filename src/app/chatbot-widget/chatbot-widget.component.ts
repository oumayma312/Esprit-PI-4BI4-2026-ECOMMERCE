import { AfterViewChecked, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { finalize } from 'rxjs/operators';
import {
  ApiHealthResponse,
  ChatbotResponse,
  ChatbotService,
  ChatbotTurn
} from '../services/chatbot.service';

interface ChatMessage {
  id: number;
  role: 'assistant' | 'user';
  content: string;
  meta?: string;
  sources?: string[];
  pending?: boolean;
}

@Component({
  selector: 'app-chatbot-widget',
  templateUrl: './chatbot-widget.component.html',
  styleUrls: ['./chatbot-widget.component.css']
})
export class ChatbotWidgetComponent implements AfterViewChecked, OnInit {
  @ViewChild('messageList') private messageListRef?: ElementRef<HTMLDivElement>;

  isOpen = false;
  isSending = false;
  draft = '';
  apiState: 'checking' | 'ready' | 'offline' = 'checking';
  apiStatusMessage = 'Connecting to the local chatbot API...';
  private nextId = 2;
  private shouldScroll = false;
  private isCheckingHealth = false;

  messages: ChatMessage[] = [
    {
      id: 1,
      role: 'assistant',
      content:
        'Hi, I am Story AI. Ask me about KPIs, products, campaigns, or any table in the data warehouse.'
    }
  ];

  constructor(private chatbot: ChatbotService) {}

  ngOnInit(): void {
    this.refreshApiStatus();
  }

  ngAfterViewChecked(): void {
    if (!this.shouldScroll || !this.messageListRef) {
      return;
    }

    const container = this.messageListRef.nativeElement;
    container.scrollTop = container.scrollHeight;
    this.shouldScroll = false;
  }

  toggleOpen(): void {
    this.isOpen = !this.isOpen;
    if (this.isOpen) {
      this.refreshApiStatus();
    }
    this.requestScroll();
  }

  close(): void {
    this.isOpen = false;
  }

  handleComposerKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  sendMessage(): void {
    const message = this.draft.trim();
    if (!message || this.isSending) {
      return;
    }

    if (this.apiState === 'offline') {
      const pendingId = this.appendMessage({
        role: 'assistant',
        content: 'Checking the chatbot API connection...',
        pending: true
      });
      this.refreshApiStatus(() => {
        if (this.apiState === 'ready') {
          this.resolvePendingMessage(pendingId, {
            answer: 'The chatbot API is back online. Please send your message again.',
            mode: 'info',
            model: 'local',
            sources: [],
            response_ms: 0
          });
          return;
        }

        this.resolvePendingMessage(pendingId, {
          answer: 'The chatbot API is still offline. Start it with `npm start` or `npm run start:api`, then try again.',
          mode: 'error',
          model: 'local',
          sources: [],
          response_ms: 0
        });
      });
      return;
    }

    const history = this.buildHistory();

    this.appendMessage({
      role: 'user',
      content: message
    });
    this.draft = '';
    this.isSending = true;

    const pendingId = this.appendMessage({
      role: 'assistant',
      content: 'Thinking...',
      pending: true
    });

    this.chatbot
      .ask(message, history)
      .pipe(finalize(() => (this.isSending = false)))
      .subscribe({
        next: response => this.resolvePendingMessage(pendingId, response),
        error: error => {
          const details = this.chatbot.describeError(error);
          this.apiState = details.kind === 'server' ? 'ready' : 'offline';
          this.apiStatusMessage = details.message;
          this.resolvePendingMessage(pendingId, {
            answer: details.message,
            mode: 'error',
            model: 'local',
            sources: [],
            response_ms: 0
          });
        }
      });
  }

  trackMessage(index: number, message: ChatMessage): number {
    return message.id;
  }

  private buildHistory(): ChatbotTurn[] {
    return this.messages
      .filter(message => !message.pending)
      .map(message => ({
        role: message.role,
        content: message.content
      }));
  }

  private appendMessage(message: Omit<ChatMessage, 'id'>): number {
    const id = this.nextId++;
    this.messages = [...this.messages, { id, ...message }];
    this.requestScroll();
    return id;
  }

  private resolvePendingMessage(pendingId: number, response: ChatbotResponse): void {
    this.messages = this.messages.map(message =>
      message.id === pendingId
        ? {
            ...message,
            content: response.answer,
            pending: false,
            meta: this.buildMeta(response),
            sources: response.sources
          }
        : message
    );
    this.requestScroll();
  }

  private buildMeta(response: ChatbotResponse): string {
    const modeLabel =
      response.mode === 'hybrid'
        ? 'Warehouse-grounded'
        : response.mode === 'gemini'
          ? 'General AI'
          : response.mode === 'info'
            ? 'Connection'
            : response.mode === 'error'
              ? 'API error'
          : 'Local analytics';
    return `${modeLabel} - ${response.response_ms} ms`;
  }

  private requestScroll(): void {
    this.shouldScroll = true;
  }

  private refreshApiStatus(onComplete?: () => void): void {
    if (this.isCheckingHealth) {
      return;
    }

    this.isCheckingHealth = true;
    this.apiState = 'checking';
    this.apiStatusMessage = 'Connecting to the local chatbot API...';

    this.chatbot.checkHealth().subscribe({
      next: response => {
        this.updateApiStatus(response);
        this.isCheckingHealth = false;
        onComplete?.();
      },
      error: error => {
        this.apiState = 'offline';
        this.apiStatusMessage = this.chatbot.describeError(error).message;
        this.isCheckingHealth = false;
        onComplete?.();
      }
    });
  }

  private updateApiStatus(response: ApiHealthResponse): void {
    if (!response.chatbot.ready) {
      this.apiState = 'offline';
      this.apiStatusMessage = 'The chatbot API responded, but the chatbot service is not ready yet.';
      return;
    }

    this.apiState = 'ready';
    this.apiStatusMessage = response.chatbot.gemini_enabled
      ? `API online. ${response.chatbot.gemini_status}`
      : 'API online. Local analytics mode is ready.';
  }
}
