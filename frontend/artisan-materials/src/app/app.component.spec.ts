import { ComponentFixture, TestBed, fakeAsync, tick, flush } from '@angular/core/testing';
import { FormsModule } from '@angular/forms';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Observable } from 'rxjs';
import { AppComponent } from './app';
import { ApiService, Material, Order, Manufacturer } from './api.service';
import { HighlightPipe } from './highlight.pipe';

describe('AppComponent', () => {
  let fixture: ComponentFixture<AppComponent>;
  let component: AppComponent;
  let apiService: any;

  const mockMaterials: Material[] = [
    { id: 1, name: 'Copper Wire 2mm', category: 'Metal', unit_price: 4.50, unit: 'meter', min_order_qty: 10, stock_quantity: 150 },
    { id: 2, name: 'Vegetable Tanned Leather', category: 'Leather', unit_price: 35.00, unit: 'sq meter', min_order_qty: 3, stock_quantity: 20 }
  ];

  const mockOrders: Order[] = [
    { id: 1, material_id: 1, quantity: 50, status: 'pending', notes: 'Urgent', material_name: 'Copper Wire 2mm' }
  ];

  const mockManufacturers: Manufacturer[] = [
    { id: 1, name: 'Marseille Copper Works', email: 'orders@marseille-copper.fr', phone: '+33', location: 'France', materials_supplied: 'Copper' }
  ];

  beforeEach(async () => {
    apiService = {
      getMaterials: vi.fn().mockReturnValue(new Observable(() => {})),
      createMaterial: vi.fn().mockReturnValue(new Observable(() => {})),
      updateMaterial: vi.fn().mockReturnValue(new Observable(() => {})),
      deleteMaterial: vi.fn().mockReturnValue(new Observable(() => {})),
      getManufacturers: vi.fn().mockReturnValue(new Observable(() => {})),
      getOrders: vi.fn().mockReturnValue(new Observable(() => {})),
      createOrder: vi.fn().mockReturnValue(new Observable(() => {})),
      updateOrderStatus: vi.fn().mockReturnValue(new Observable(() => {})),
      deleteOrder: vi.fn().mockReturnValue(new Observable(() => {})),
      streamChat: vi.fn().mockReturnValue({ close: vi.fn() }),
      approveChat: vi.fn().mockReturnValue(new Observable(() => {})),
      rejectChat: vi.fn().mockReturnValue(new Observable(() => {})),
      editChat: vi.fn().mockReturnValue(new Observable(() => {})),
    };

    await TestBed.configureTestingModule({
      imports: [AppComponent, FormsModule, HighlightPipe],
      providers: [{ provide: ApiService, useValue: apiService }]
    }).compileComponents();

    fixture = TestBed.createComponent(AppComponent);
    component = fixture.componentInstance;
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  describe('activeTab', () => {
    it('should default to materials', () => {
      expect(component.activeTab()).toBe('materials');
    });

    it('should switch to orders', () => {
      component.activeTab.set('orders');
      expect(component.activeTab()).toBe('orders');
    });

    it('should switch to chat', () => {
      component.activeTab.set('chat');
      expect(component.activeTab()).toBe('chat');
    });
  });

  describe('getCategoryClass', () => {
    it('should return metal for Metal category', () => {
      expect(component.getCategoryClass('Metal')).toBe('metal');
    });

    it('should return leather for Leather category', () => {
      expect(component.getCategoryClass('Leather')).toBe('leather');
    });

    it('should return textile for Textile category', () => {
      expect(component.getCategoryClass('Textile')).toBe('textile');
    });

    it('should return textile for fabric category', () => {
      expect(component.getCategoryClass('Fabric')).toBe('textile');
    });

    it('should return textile for thread category', () => {
      expect(component.getCategoryClass('Thread')).toBe('textile');
    });

    it('should return other for unknown category', () => {
      expect(component.getCategoryClass('Wood')).toBe('other');
    });
  });

  describe('getMaterialName', () => {
    it('should return material name by id', () => {
      component.materials.set(mockMaterials);
      expect(component.getMaterialName(1)).toBe('Copper Wire 2mm');
    });

    it('should return ID number for missing id', () => {
      expect(component.getMaterialName(999)).toBe('ID 999');
    });

    it('should return Unknown for undefined id', () => {
      expect(component.getMaterialName(undefined)).toBe('Unknown');
    });

    it('should return Unknown for null id', () => {
      expect(component.getMaterialName(null)).toBe('Unknown');
    });
  });

  describe('statusColor', () => {
    it('should return color for pending', () => {
      expect(component.statusColor('pending')).toBeTruthy();
    });

    it('should return color for approved', () => {
      expect(component.statusColor('approved')).toBeTruthy();
    });

    it('should return color for completed', () => {
      expect(component.statusColor('completed')).toBeTruthy();
    });

    it('should return color for cancelled', () => {
      expect(component.statusColor('cancelled')).toBeTruthy();
    });
  });

  describe('toast', () => {
    it('should set toast properties on error', () => {
      component.showToastMessage('Test message', 'error');
      expect(component.toastMessage).toBe('Test message');
      expect(component.toastType).toBe('error');
    });

    it('should set toast properties on success', () => {
      component.showToastMessage('Success!', 'success');
      expect(component.toastType).toBe('success');
    });

    it('should set toast properties on info', () => {
      component.showToastMessage('Info', 'info');
      expect(component.toastType).toBe('info');
    });
  });

  describe('material form', () => {
    it('should initialize form with defaults', () => {
      expect(component.materialForm.stock_quantity).toBe(0);
      expect(component.materialForm.min_order_qty).toBe(1);
      expect(component.materialForm.unit).toBe('meter');
    });

    it('should open form with existing material data', () => {
      component.openMaterialForm(mockMaterials[0]);
      expect(component.editingMaterial()?.id).toBe(1);
      expect(component.materialForm.name).toBe('Copper Wire 2mm');
      expect(component.materialForm.stock_quantity).toBe(150);
      expect(component.showMaterialForm()).toBe(true);
    });

    it('should open new form with empty data', () => {
      component.openMaterialForm();
      expect(component.editingMaterial()).toBeNull();
      expect(component.materialForm.name).toBe('');
      expect(component.materialForm.stock_quantity).toBe(0);
      expect(component.showMaterialForm()).toBe(true);
    });

    it('should close material form', () => {
      component.openMaterialForm();
      component.closeMaterialForm();
      expect(component.showMaterialForm()).toBe(false);
      expect(component.editingMaterial()).toBeNull();
    });
  });

  describe('order form', () => {
    it('should open order form with defaults', () => {
      component.openOrderForm();
      expect(component.showOrderForm()).toBe(true);
      expect(component.orderForm.quantity).toBe(1);
    });
  });

  describe('chat message handling', () => {
    it('should generate timestamp', () => {
      const ts = component.getTimestamp();
      expect(ts).toBeTruthy();
      expect(ts.length).toBeGreaterThan(0);
    });

    it('should add user message with timestamp when sending chat', () => {
      component.chatMessages.set([]);
      component.chatInput = 'Hello';
      component.sendChat();
      const msgs = component.chatMessages();
      expect(msgs.length).toBe(2); // user message + streaming assistant
      expect(msgs[0].role).toBe('user');
      expect(msgs[0].content).toBe('Hello');
      expect(msgs[0].timestamp).toBeTruthy();
      expect(msgs[1].isStreaming).toBe(true);
    });

    it('should clear input after sending', () => {
      component.chatInput = 'Test message';
      component.sendChat();
      expect(component.chatInput).toBe('');
    });

    it('should not send empty message', () => {
      component.chatMessages.set([]);
      component.chatInput = '';
      component.sendChat();
      expect(component.chatMessages().length).toBe(0);
    });
  });

  describe('prefill data', () => {
    it('should have prefillData signal', () => {
      expect(component.prefillData()).toBeNull();
    });

    it('should set prefill data with materials array', () => {
      component.prefillData.set({ materials: [{ material_id: 1, quantity: 50 }, { material_id: 2, quantity: 20 }], notes: 'Test' });
      expect(component.prefillData()?.materials[0].material_id).toBe(1);
      expect(component.prefillData()?.materials[0].quantity).toBe(50);
      expect(component.prefillData()?.materials[1].material_id).toBe(2);
      expect(component.prefillData()?.materials[1].quantity).toBe(20);
      expect(component.prefillData()?.notes).toBe('Test');
    });
  });

  describe('stock_quantity in form', () => {
    it('should include stock_quantity in material form', () => {
      component.openMaterialForm({
        id: 1, name: 'Test', category: 'Test', unit_price: 1.0,
        unit: 'piece', min_order_qty: 1, stock_quantity: 42
      });
      expect(component.materialForm.stock_quantity).toBe(42);
    });
  });

  describe('approveThread', () => {
    it('should call approve endpoint', () => {
      component.chatMessages.set([{ role: 'assistant', content: 'test', pending: true, threadId: 'thread-1' }]);
      apiService.approveChat.mockReturnValue(new Observable(() => {}));
      component.approveThread('thread-1');
      expect(apiService.approveChat).toHaveBeenCalledWith('thread-1');
    });

    it('should clear interrupt and prefill on approve', () => {
      component.currentInterrupt.set({ name: 'test', arguments: {}, description: '' });
      component.prefillData.set({ materials: [{ material_id: 1, quantity: 10 }] });
      component.approveThread('thread-1');
      expect(component.currentInterrupt()).toBeNull();
      expect(component.prefillData()).toBeNull();
    });
  });

  describe('rejectThread', () => {
    it('should call reject endpoint', () => {
      component.chatMessages.set([{ role: 'assistant', content: 'test', pending: true, threadId: 'thread-1' }]);
      apiService.rejectChat.mockReturnValue(new Observable(() => {}));
      component.rejectThread('thread-1');
      expect(apiService.rejectChat).toHaveBeenCalledWith('thread-1');
    });

    it('should clear interrupt and prefill on reject', () => {
      component.currentInterrupt.set({ name: 'test', arguments: {}, description: '' });
      component.prefillData.set({ materials: [{ material_id: 1, quantity: 10 }] });
      component.rejectThread('thread-1');
      expect(component.currentInterrupt()).toBeNull();
      expect(component.prefillData()).toBeNull();
    });
  });

  describe('editThread', () => {
    it('should call edit endpoint with edited values', () => {
      component.chatMessages.set([{ role: 'assistant', content: 'test', pending: true, threadId: 'thread-1' }]);
      component.currentInterrupt.set({ name: 'create_order', arguments: {}, description: '' });
      component.editItems = [{ material_id: 2, quantity: 25 }];
      component.editNotes = 'Edited note';
      apiService.editChat.mockReturnValue(new Observable(() => {}));
      component.editThread('thread-1');
      expect(apiService.editChat).toHaveBeenCalledWith('thread-1', 'create_order', {
        materials: [{ material_id: 2, quantity: 25 }],
        notes: 'Edited note'
      });
    });

    it('should show error for missing material in edit', () => {
      component.chatMessages.set([{ role: 'assistant', content: 'test', pending: true, threadId: 'thread-1' }]);
      component.currentInterrupt.set({ name: 'create_order', arguments: {}, description: '' });
      component.editItems = [{ material_id: null, quantity: 10 }];
      component.editThread('thread-1');
      expect(component.toastMessage).toContain('Please select a material for all items');
    });

    it('should show error for quantity below 1 in edit', () => {
      component.chatMessages.set([{ role: 'assistant', content: 'test', pending: true, threadId: 'thread-1' }]);
      component.currentInterrupt.set({ name: 'create_order', arguments: {}, description: '' });
      component.editItems = [{ material_id: 1, quantity: 0 }];
      component.editThread('thread-1');
      expect(component.toastMessage).toContain('Quantity must be at least 1 for all items');
    });

    it('should clear interrupt and prefill on edit', () => {
      component.currentInterrupt.set({ name: 'test', arguments: {}, description: '' });
      component.prefillData.set({ materials: [{ material_id: 1, quantity: 10 }] });
      component.editItems = [{ material_id: 1, quantity: 1 }];
      apiService.editChat.mockReturnValue(new Observable(() => {}));
      component.editThread('thread-1');
      expect(component.currentInterrupt()).toBeNull();
      expect(component.prefillData()).toBeNull();
    });
  });

  describe('saveMaterial validation', () => {
    it('should show error for missing name', () => {
      component.materialForm = { name: '', category: 'Test', unit_price: 1.0, unit: 'piece', min_order_qty: 1, stock_quantity: 0, manufacturer_id: undefined };
      component.saveMaterial();
      expect(component.toastMessage).toContain('Name and category are required');
    });

    it('should show error for missing category', () => {
      component.materialForm = { name: 'Test', category: '', unit_price: 1.0, unit: 'piece', min_order_qty: 1, stock_quantity: 0, manufacturer_id: undefined };
      component.saveMaterial();
      expect(component.toastMessage).toContain('Name and category are required');
    });
  });

  describe('saveOrder validation', () => {
    it('should show error for missing material', () => {
      component.orderForm = { material_id: undefined, quantity: 1, notes: '' };
      component.saveOrder();
      expect(component.toastMessage).toContain('Select a material');
    });
  });
});
