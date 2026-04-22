import { Component, signal, computed, OnInit, ElementRef, ViewChild, AfterViewInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { firstValueFrom } from 'rxjs';
import { ApiService, Material, Order, Manufacturer, InterruptData, PrefillData, Customer, CustomerFeatures, SegmentStat } from './api.service';
import { HighlightPipe } from './highlight.pipe';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  pending?: boolean;
  threadId?: string;
  isStreaming?: boolean;
  timestamp?: string;
}

interface EditItem {
  material_id: number | null;
  quantity: number;
}

@Component({
  selector: 'app-root',
  imports: [FormsModule, CommonModule, HighlightPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class AppComponent implements OnInit, AfterViewInit {
  readonly api = ApiService;
  activeTab = signal<'materials' | 'orders' | 'customers' | 'chat'>('materials');

  @ViewChild('chatMessagesRef') chatMessagesRef!: ElementRef;
  @ViewChild('resultRef') resultRef!: ElementRef;

  // Materials state
  materials = signal<Material[]>([]);
  editingMaterial = signal<Material | null>(null);
  showMaterialForm = signal(false);
  materialForm = { name: '', category: '', unit_price: 0, unit: 'meter', min_order_qty: 1, stock_quantity: 0, manufacturer_id: undefined as number | undefined };

  // Manufacturers state
  manufacturers = signal<Manufacturer[]>([]);

  // Orders state
  orders = signal<Order[]>([]);
  showOrderForm = signal(false);
  orderForm = { material_id: undefined as number | undefined, quantity: 1, notes: '' };

  // Chat state
  chatMessages = signal<ChatMessage[]>([]);
  chatInput = '';
  currentThreadId = '';
  chatLoading = false;

  // Streaming state
  isStreaming = false;
  streamingContent = '';
  currentEventSource: EventSource | null = null;

  // HITL Edit Panel
  currentInterrupt = signal<InterruptData | null>(null);
  editItems: EditItem[] = [{ material_id: null, quantity: 1 }];
  editNotes = '';
  prefillData = signal<PrefillData | null>(null);

  // Toast notification
  showToast = signal(false);
  toastMessage = '';
  toastType = 'error';

  // Customers state
  customers = signal<Customer[]>([]);
  segmentStats = signal<SegmentStat[]>([]);
  searchQuery = '';
  segmentFilter: number | null = null;
  riskFilter: string | null = null;
  selectedCustomer: Customer | null = null;
  pcaPlotUrl: string = '';

  // Pagination state
  currentPage = signal(1);
  pageSize = signal(50);

  // Manual prediction form
  showPredictForm = false;
  predictionForm: CustomerFeatures = {
    recency: 120, frequency: 2, monetary_total: 250,
    monetary_trend: 1, product_diversity: 3, channel_diversity: 1,
    lifetime_days: 365, purchase_velocity: 1.5, avg_price: 80, channel: ''
  };
  predictionResult = signal<any>(null);
  predictionError = '';
  predictionLoading = false;

  // Language toggle
  lang = signal<'fr' | 'en'>('fr');

  constructor(private apiService: ApiService) {
    const saved = localStorage.getItem('app-lang');
    if (saved === 'en') this.lang.set('en');
  }

  // --- Translations ---
  t(key: string): string {
    const lang = this.lang();
    const dict: Record<string, Record<string, string>> = {
      'header.subtitle': {
        fr: 'Matériaux, commandes, clients & assistant IA pour l\'atelier',
        en: 'Materials, orders, customers & AI assistant for the workshop'
      },
      'tabs.materials': { fr: 'Matériaux', en: 'Materials' },
      'tabs.orders': { fr: 'Commandes', en: 'Orders' },
      'tabs.customers': { fr: 'Clients', en: 'Customers' },
      'tabs.chat': { fr: 'Assistant IA', en: 'AI Assistant' },
      'materials.title': { fr: 'Inventaire', en: 'Inventory' },
      'materials.add': { fr: '+ Ajouter Matériau', en: '+ Add Material' },
      'materials.empty': { fr: 'Aucun matériau. Cliquez sur "+ Ajouter Matériau" pour commencer.', en: 'No materials yet. Click "+ Add Material" to start your inventory.' },
      'orders.title': { fr: 'Commandes', en: 'Orders' },
      'orders.add': { fr: '+ Nouvelle Commande', en: '+ New Order' },
      'orders.empty': { fr: 'Aucune commande. Utilisez l\'assistant IA ou créez-en une manuellement.', en: 'No orders yet. Use the AI assistant or create one manually.' },
      'customers.title': { fr: 'Segments Clients', en: 'Customer Segments' },
      'customers.predict': { fr: 'Prédire le Risque Client', en: 'Predict Customer Risk' },
      'customers.predict_btn': { fr: 'Prédire', en: 'Predict' },
      'customers.customer': { fr: 'Client', en: 'Customer' },
      'customers.code': { fr: 'Code', en: 'Code' },
      'customers.segment': { fr: 'Segment', en: 'Segment' },
      'customers.churn_prob': { fr: 'Prob Churn', en: 'Churn Prob' },
      'customers.risk': { fr: 'Risque', en: 'Risk' },
      'customers.status': { fr: 'Statut', en: 'Status' },
      'customers.actions': { fr: 'Actions', en: 'Actions' },
      'customers.details': { fr: 'Détails', en: 'Details' },
      'customers.close': { fr: 'Fermer', en: 'Close' },
      'customers.recency': { fr: 'Récence (jours)', en: 'Recency (days)' },
      'customers.frequency': { fr: 'Fréquence', en: 'Frequency' },
      'customers.monetary': { fr: 'Montant Total', en: 'Monetary Total' },
      'customers.trend': { fr: 'Tendance Monétaire', en: 'Monetary Trend' },
      'customers.product_div': { fr: 'Diversité Produits', en: 'Product Diversity' },
      'customers.channel_div': { fr: 'Diversité Canaux', en: 'Channel Diversity' },
      'customers.lifetime': { fr: 'Durée (jours)', en: 'Lifetime (days)' },
      'customers.velocity': { fr: 'Vitesse Achat', en: 'Purchase Velocity' },
      'customers.avg_price': { fr: 'Prix Moyen', en: 'Avg Price' },
      'customers.channel': { fr: 'Canal', en: 'Channel' },
      'customers.baseline': { fr: '-- Baseline --', en: '-- Baseline --' },
      'customers.recommendation': { fr: 'Recommandation', en: 'Recommendation' },
      'customers.search_placeholder': { fr: 'Rechercher des clients...', en: 'Search customers...' },
      'customers.all_segments': { fr: 'Tous les Segments', en: 'All Segments' },
      'customers.all_risk': { fr: 'Tous les Niveaux', en: 'All Risk Levels' },
      'customers.no_results': { fr: 'Aucun client trouvé.', en: 'No customers found.' },
      'customers.search_btn': { fr: 'Rechercher', en: 'Search' },
      'customers.segments.vip': { fr: 'VIP', en: 'VIP' },
      'customers.segments.lost': { fr: 'Perdu', en: 'Lost' },
      'customers.segments.atrisk': { fr: 'A risque', en: 'At-risk' },
      'customers.segments.new': { fr: 'Nouveau', en: 'New' },
      'pagination.page_size': { fr: 'Lignes par page', en: 'Rows per page' },
      'pagination.of': { fr: 'sur', en: 'of' },
      'pagination.predict_hint': { fr: 'Ajustez les valeurs pour prédire le risque de churn', en: 'Adjust values to predict churn risk' },
      'pagination.predict_btn': { fr: '▶ Prédire', en: '▶ Predict' },
      'form.cancel': { fr: 'Annuler', en: 'Cancel' },
      'form.save': { fr: 'Enregistrer', en: 'Save' },
      'order.new': { fr: 'Nouvelle Commande', en: 'New Order' },
      'order.create': { fr: 'Créer Commande', en: 'Create Order' },
      'order.select_material': { fr: '-- Sélectionner --', en: '-- Select --' },
      'order.notes': { fr: 'Notes', en: 'Notes' },
      'order.notes_placeholder': { fr: 'Notes optionnelles', en: 'Optional notes' },
      'material.edit': { fr: 'Modifier', en: 'Edit' },
      'material.new': { fr: 'Nouveau', en: 'New' },
      'material.name': { fr: 'Nom', en: 'Name' },
      'material.name_placeholder': { fr: 'ex. Fil de cuivre 2mm', en: 'e.g. Copper Wire 2mm' },
      'material.category': { fr: 'Catégorie', en: 'Category' },
      'material.category_placeholder': { fr: 'ex. Metal, Cuir, Textile', en: 'e.g. Metal, Leather, Textile' },
      'material.price': { fr: 'Prix (par unité)', en: 'Price (per unit)' },
      'material.unit': { fr: 'Unité', en: 'Unit' },
      'material.min_order': { fr: 'Cmd Min', en: 'Min Order Qty' },
      'material.in_stock': { fr: 'En Stock', en: 'In Stock' },
      'material.manufacturer': { fr: 'Fabricant', en: 'Manufacturer' },
      'material.none': { fr: '-- Aucun --', en: '-- None --' },
      'material.quantity': { fr: 'Quantité', en: 'Quantity' },
      'chat.welcome_title': { fr: 'Assistant IA Matériaux', en: 'AI Materials Assistant' },
      'chat.welcome_desc': { fr: 'Demandez-moi de rechercher des matériaux, passer des commandes, ou trouver des fabricants. Je peux pré-remplir les commandes pour révision.', en: 'Ask me to search materials, place orders, or find manufacturers. I can pre-fill orders for you to review.' },
      'chat.placeholder': { fr: 'Demandez sur les matériaux, commandes ou fabricants...', en: 'Ask about materials, orders, or manufacturers...' },
      'chat.send': { fr: 'Envoyer', en: 'Send' },
      'chat.suggestions.1': { fr: 'J\'ai besoin de 50m de fil de cuivre', en: 'Need 50m copper wire' },
      'chat.suggestions.2': { fr: 'Montrez les matériaux Metal', en: 'Show metal materials' },
      'chat.suggestions.3': { fr: 'Recherchez fournisseurs cuir Italie', en: 'Search leather suppliers' },
      'chat.suggestions.4': { fr: 'Quelles commandes en attente?', en: 'Pending orders' },
      'prediction.churn_prob': { fr: 'Probabilité Churn', en: 'Churn Probability' },
      'prediction.status': { fr: 'Statut', en: 'Status' },
      'prediction.segment': { fr: 'Segment', en: 'Segment' },
      'prediction.risk': { fr: 'Niveau Risque', en: 'Risk Level' },
      'prediction.recommendation': { fr: 'Recommandation', en: 'Recommendation' },
    };
    return (dict[key]?.[lang] ?? dict[key]?.['fr'] ?? key);
  }

  ngOnInit() {
    this.loadMaterials();
    this.loadManufacturers();
    this.loadOrders();
    this.loadSegmentStats();
    this.loadPCAPlot();
    this.loadCustomers();
  }

  ngAfterViewInit() {
    this.scrollToBottom();
  }

  private scrollToBottom() {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const el = this.chatMessagesRef?.nativeElement;
        if (el) {
          el.scrollTop = el.scrollHeight;
        }
      });
    });
  }

  // --- Toast ---
  showToastMessage(message: string, type: 'error' | 'success' | 'info') {
    this.toastMessage = message;
    this.toastType = type;
    this.showToast.set(true);
    setTimeout(() => {
      this.showToast.set(false);
    }, 4000);
  }

  // --- Helpers ---
  getCategoryClass(category: string): string {
    const c = category.toLowerCase();
    if (c.includes('metal')) return 'metal';
    if (c.includes('leather')) return 'leather';
    if (c.includes('textile') || c.includes('fabric') || c.includes('thread')) return 'textile';
    return 'other';
  }

  getMaterialName(id?: number | null): string {
    if (!id) return 'Unknown';
    const m = this.materials().find(x => x.id === id);
    return m?.name || `ID ${id}`;
  }

  getTimestamp(): string {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  statusColor(status: string): string {
    const colors: Record<string, string> = {
      pending: '#c4883a', approved: '#b87333', completed: '#4a8c6a', cancelled: '#b85450'
    };
    return colors[status] || '#666';
  }

  // --- Materials ---
  loadMaterials() {
    this.apiService.getMaterials().subscribe({
      next: (data) => this.materials.set(data),
      error: (e) => this.showToastMessage(`Error loading materials: ${e.message}`, 'error')
    });
  }

  openMaterialForm(m?: Material) {
    this.editingMaterial.set(m || null);
    if (m) {
      this.materialForm = {
        name: m.name,
        category: m.category,
        unit_price: m.unit_price,
        unit: m.unit,
        min_order_qty: m.min_order_qty,
        stock_quantity: m.stock_quantity || 0,
        manufacturer_id: m.manufacturer_id
      };
    } else {
      this.materialForm = { name: '', category: '', unit_price: 0, unit: 'meter', min_order_qty: 1, stock_quantity: 0, manufacturer_id: undefined };
    }
    this.showMaterialForm.set(true);
  }

  saveMaterial() {
    const f = this.materialForm;
    if (!f.name || !f.category) {
      this.showToastMessage('Name and category are required', 'error');
      return;
    }
    const payload: Material = {
      name: f.name, category: f.category, unit_price: f.unit_price,
      unit: f.unit, min_order_qty: f.min_order_qty, stock_quantity: f.stock_quantity, manufacturer_id: f.manufacturer_id
    };
    const editId = this.editingMaterial()?.id;
    if (editId) {
      this.apiService.updateMaterial(editId, payload).subscribe({
        next: () => { this.loadMaterials(); this.closeMaterialForm(); this.showToastMessage('Material updated', 'success'); },
        error: (e) => this.showToastMessage(`Error: ${e.message}`, 'error')
      });
    } else {
      this.apiService.createMaterial(payload).subscribe({
        next: () => { this.loadMaterials(); this.closeMaterialForm(); this.showToastMessage('Material added', 'success'); },
        error: (e) => this.showToastMessage(`Error: ${e.message}`, 'error')
      });
    }
  }

  deleteMaterial(id: number) {
    if (!confirm('Delete this material?')) return;
    this.apiService.deleteMaterial(id).subscribe({
      next: () => this.loadMaterials(),
      error: (e) => this.showToastMessage(`Error: ${e.message}`, 'error')
    });
  }

  closeMaterialForm() {
    this.showMaterialForm.set(false);
    this.editingMaterial.set(null);
  }

  // --- Manufacturers ---
  loadManufacturers() {
    this.apiService.getManufacturers().subscribe({
      next: (data) => this.manufacturers.set(data)
    });
  }

  // --- Orders ---
  loadOrders() {
    this.apiService.getOrders().subscribe({
      next: (data) => this.orders.set(data)
    });
  }

  openOrderForm() {
    this.orderForm = { material_id: undefined, quantity: 1, notes: '' };
    this.showOrderForm.set(true);
  }

  saveOrder() {
    const f = this.orderForm;
    if (!f.material_id) { this.showToastMessage('Select a material', 'error'); return; }
    this.apiService.createOrder({ material_id: f.material_id, quantity: f.quantity, notes: f.notes, status: 'pending' }).subscribe({
      next: () => { this.loadOrders(); this.showOrderForm.set(false); this.showToastMessage('Order created', 'success'); },
      error: (e) => this.showToastMessage(`Error: ${e.message}`, 'error')
    });
  }

  updateOrderStatus(id: number, status: string) {
    this.apiService.updateOrderStatus(id, status).subscribe({
      next: () => this.loadOrders(),
      error: (e) => this.showToastMessage(`Error: ${e.message}`, 'error')
    });
  }

  deleteOrder(id: number) {
    if (!confirm('Delete this order?')) return;
    this.apiService.deleteOrder(id).subscribe({
      next: () => this.loadOrders(),
      error: (e) => this.showToastMessage(`Error: ${e.message}`, 'error')
    });
  }

  // --- Chat ---
  sendChat() {
    const msg = this.chatInput.trim();
    if (!msg) return;

    this.chatInput = '';
    this.isStreaming = true;
    this.streamingContent = '';

    // Add user message with timestamp
    this.chatMessages.update(m => [...m, {
      role: 'user',
      content: msg,
      timestamp: this.getTimestamp()
    }]);

    // Add streaming message bubble
    const streamMsg: ChatMessage = {
      role: 'assistant',
      content: '',
      isStreaming: true
    };
    this.chatMessages.update(m => [...m, streamMsg]);

    // Scroll after Angular has rendered the new messages
    this.scrollToBottom();

    const threadId = this.currentThreadId || undefined;
    this.currentEventSource = this.apiService.streamChat(msg, threadId);

    this.currentEventSource.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === 'thread_id') {
          this.currentThreadId = data.thread_id;
        } else if (data.type === 'token') {
          this.streamingContent += data.content;
          // Update last message (the streaming one)
          this.chatMessages.update(m => {
            const last = m[m.length - 1];
            if (last.isStreaming) {
              return [...m.slice(0, -1), { ...last, content: this.streamingContent }];
            }
            return m;
          });
          this.scrollToBottom();
        } else if (data.type === 'interrupt') {
          this.currentInterrupt.set(data.data[0] || null);
          // Auto-prefill from interrupt data
          if (data.prefill) {
            this.prefillData.set(data.prefill);
            // Prefer multi-material format, fall back to single material
            if (data.prefill.materials && data.prefill.materials.length > 0) {
              this.editItems = data.prefill.materials.map((m: { material_id: number | null; quantity: number }) => ({
                material_id: m.material_id ?? null,
                quantity: m.quantity || 1
              }));
            } else if (data.prefill.material_id !== undefined) {
              this.editItems = [{ material_id: data.prefill.material_id, quantity: data.prefill.quantity || 1 }];
            }
            this.editNotes = data.prefill.notes || '';
          }
          this.isStreaming = false;
          // Set the last message as pending approval
          this.chatMessages.update(m => {
            const last = m[m.length - 1];
            if (last.isStreaming) {
              return [...m.slice(0, -1), {
                ...last,
                isStreaming: false,
                pending: true,
                threadId: this.currentThreadId,
                timestamp: this.getTimestamp()
              }];
            }
            return m;
          });
          this.scrollToBottom();
          this.currentEventSource?.close();
        } else if (data.type === 'complete') {
          this.isStreaming = false;
          // Finalize streaming message
          this.chatMessages.update(m => {
            const last = m[m.length - 1];
            if (last.isStreaming) {
              return [...m.slice(0, -1), { ...last, isStreaming: false, content: this.streamingContent, timestamp: this.getTimestamp() }];
            }
            return m;
          });
          this.scrollToBottom();
          this.currentEventSource?.close();
        } else if (data.type === 'error') {
          this.showToastMessage(`Error: ${data.message}`, 'error');
          this.isStreaming = false;
          this.currentEventSource?.close();
          // Remove streaming message
          this.chatMessages.update(m => m.filter(msg => !msg.isStreaming));
        }
      } catch (e) {
        // Ignore parse errors (e.g., empty SSE messages)
      }
    };

    this.currentEventSource.onerror = () => {
      this.showToastMessage('Connection error', 'error');
      this.isStreaming = false;
      this.currentEventSource?.close();
    };
  }

  addMaterialItem() {
    this.editItems.push({ material_id: null, quantity: 1 });
  }

  removeMaterialItem(index: number) {
    if (this.editItems.length > 1) {
      this.editItems.splice(index, 1);
    }
  }

  approveThread(threadId: string) {
    // Remove pending message
    this.chatMessages.update(m => m.filter(msg => !(msg.threadId === threadId && msg.pending)));
    this.currentInterrupt.set(null);
    this.prefillData.set(null);

    this.apiService.approveChat(threadId).subscribe({
      next: (resp: any) => {
        this.chatMessages.update(m => [...m, {
          role: 'assistant',
          content: resp.response || 'Decision recorded.',
          timestamp: this.getTimestamp()
        }]);
        this.showToastMessage('Order approved', 'success');
        this.scrollToBottom();
      },
      error: (e) => this.showToastMessage(`Approve error: ${e.message}`, 'error')
    });
  }

  rejectThread(threadId: string) {
    // Remove pending message
    this.chatMessages.update(m => m.filter(msg => !(msg.threadId === threadId && msg.pending)));
    this.currentInterrupt.set(null);
    this.prefillData.set(null);

    this.apiService.rejectChat(threadId).subscribe({
      next: (resp: any) => {
        this.chatMessages.update(m => [...m, {
          role: 'assistant',
          content: resp.response || 'Decision recorded.',
          timestamp: this.getTimestamp()
        }]);
        this.showToastMessage('Order rejected', 'info');
        this.scrollToBottom();
      },
      error: (e) => this.showToastMessage(`Reject error: ${e.message}`, 'error')
    });
  }

  editThread(threadId: string) {
    const interrupt = this.currentInterrupt();
    if (!interrupt) return;

    // Validate all items
    for (const item of this.editItems) {
      if (item.material_id === null || item.material_id === undefined) {
        this.showToastMessage('Please select a material for all items', 'error');
        return;
      }
      if (item.quantity < 1) {
        this.showToastMessage('Quantity must be at least 1 for all items', 'error');
        return;
      }
    }

    // Remove pending message
    this.chatMessages.update(m => m.filter(msg => !(msg.threadId === threadId && msg.pending)));
    this.currentInterrupt.set(null);
    this.prefillData.set(null);

    this.apiService.editChat(threadId, interrupt.name, {
      materials: this.editItems.map(item => ({
        material_id: Number(item.material_id),
        quantity: item.quantity
      })),
      notes: this.editNotes
    }).subscribe({
      next: (resp: any) => {
        this.chatMessages.update(m => [...m, {
          role: 'assistant',
          content: resp.response || 'Decision recorded.',
          timestamp: this.getTimestamp()
        }]);
        this.showToastMessage('Order created with edited values', 'success');
        this.scrollToBottom();
      },
      error: (e) => this.showToastMessage(`Edit error: ${e.message}`, 'error')
    });
  }

  // --- Customers ---
  totalPages = computed(() => {
    const total = this._totalCustomersVal || 0;
    return Math.ceil(total / this.pageSize());
  });

  async loadCustomers() {
    const offset = (this.currentPage() - 1) * this.pageSize();
    try {
      const [customersData, countData] = await Promise.all([
        firstValueFrom(this.apiService.listCustomers({
          segment: this.segmentFilter,
          risk_level: this.riskFilter || undefined,
          search: this.searchQuery || undefined,
          limit: this.pageSize(),
          offset: offset
        })),
        firstValueFrom(this.apiService.getCustomerCount({
          segment: this.segmentFilter,
          risk_level: this.riskFilter || undefined,
          search: this.searchQuery || undefined
        }))
      ]);
      this.customers.set(customersData || []);
      this._totalCustomersVal = countData?.count || 0;
    } catch (e: any) {
      this.showToastMessage(`Error loading customers: ${e.message}`, 'error');
    }
  }

  goToPage(n: number) {
    this.currentPage.set(n);
    this.loadCustomers();
  }

  prevPage() {
    if (this.currentPage() > 1) {
      this.currentPage.set(this.currentPage() - 1);
      this.loadCustomers();
    }
  }

  nextPage() {
    const total = this.totalPages();
    if (this.currentPage() < total) {
      this.currentPage.set(this.currentPage() + 1);
      this.loadCustomers();
    }
  }

  onPageSizeChange(newSize: number) {
    this.pageSize.set(newSize);
    this.currentPage.set(1);
    this.loadCustomers();
  }

  loadSegmentStats() {
    this.apiService.getSegmentStats().subscribe({
      next: (data) => this.segmentStats.set(data),
      error: () => this.segmentStats.set([])
    });
  }

  loadPCAPlot() {
    this.apiService.getPCAPlot().subscribe({
      next: (blob: Blob) => {
        this.pcaPlotUrl = URL.createObjectURL(blob);
      },
      error: () => { this.pcaPlotUrl = ''; }
    });
  }

  viewCustomer(id: number) {
    this.apiService.getCustomer(id).subscribe({
      next: (data) => this.selectedCustomer = data,
      error: (e) => this.showToastMessage(`Error: ${e.message}`, 'error')
    });
  }

  closeDetail() {
    this.selectedCustomer = null;
  }

  // --- Manual Prediction ---
  runPrediction() {
    console.log('[PREDICT] Starting prediction with form:', this.predictionForm);
    this.predictionError = '';
    this.predictionResult.set(null);
    this.predictionLoading = true;
    this.apiService.predictChurn(this.predictionForm).subscribe({
      next: (data) => {
        console.log('[PREDICT] Result:', data);
        this.predictionResult.set(data);
        this.predictionLoading = false;
        setTimeout(() => {
          this.resultRef?.nativeElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
      },
      error: (e) => {
        console.error('[PREDICT] Error:', e);
        if (e.status === 0) {
          this.predictionError = 'Backend non joignable. Verifiez que le serveur tourne sur le port 8000.';
        } else if (e.status === 408 || e.code === 'TIMEOUT' || e.code === 23) {
          this.predictionError = 'Delai depasse (15s). Le backend repond pas.';
        } else {
          this.predictionError = e.error?.detail || `Erreur: ${e.message || e}`;
        }
        this.predictionLoading = false;
      }
    });
  }

  // --- Language ---
  toggleLanguage() {
    const next = this.lang() === 'fr' ? 'en' : 'fr';
    this.lang.set(next);
    localStorage.setItem('app-lang', next);
  }

  // --- Pagination helpers ---
  Math = Math;

  get _totalCustomers(): number {
    return this._totalCustomersVal;
  }

  getPageNumbers(): (number | string)[] {
    const total = this.totalPages();
    const current = this.currentPage();
    if (total <= 7) {
      return Array.from({ length: total }, (_, i) => i + 1);
    }
    const pages: (number | string)[] = [];
    pages.push(1);
    if (current > 3) {
      pages.push('...');
    }
    const start = Math.max(2, current - 2);
    const end = Math.min(total - 1, current + 2);
    for (let i = start; i <= end; i++) {
      pages.push(i);
    }
    if (current < total - 2) {
      pages.push('...');
    }
    pages.push(total);
    return pages;
  }

  private _totalCustomersVal = 0;

  // --- Risk helpers ---
  getRiskClass(risk: string): string {
    if (risk === 'Eleve') return 'risk-bad';
    if (risk === 'Modere') return 'risk-warn';
    return 'risk-good';
  }

  churnLabel(pred: string): string {
    return pred === '1' ? 'Churné' : 'Actif';
  }

  getRiskWidth(prob: number): number {
    return Math.round(prob * 100);
  }

  getSegmentClass(seg: number): string {
    return `segment-${seg}`;
  }
}
