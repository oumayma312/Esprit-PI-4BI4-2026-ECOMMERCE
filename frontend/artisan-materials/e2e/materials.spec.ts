import { test, expect } from '@playwright/test';

test.describe('Materials Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('.header h1', { timeout: 10000 });
  });

  test('should have correct page title and header', async ({ page }) => {
    await expect(page.locator('.header h1')).toHaveText('Atelier Materials');
    await expect(page.locator('.subtitle')).toContainText('Materials, orders & AI assistant for the workshop');
  });

  test('should have three tabs', async ({ page }) => {
    const tabs = page.locator('.tabs button');
    await expect(tabs).toHaveCount(3);
    await expect(tabs.nth(0)).toHaveText('Materials');
    await expect(tabs.nth(1)).toHaveText('Orders');
    await expect(tabs.nth(2)).toHaveText('AI Assistant');
  });

  test('should default to materials tab', async ({ page }) => {
    await expect(page.locator('.toolbar h2')).toHaveText('Inventory');
  });

  test('should show add material button', async ({ page }) => {
    await expect(page.locator('.btn-primary')).toContainText('+ Add Material');
  });

  test('should show empty state when no materials', async ({ page }) => {
    await expect(page.locator('.empty')).toBeVisible();
  });

  test('should switch to orders tab', async ({ page }) => {
    await page.locator('.tabs button').nth(1).click();
    await expect(page.locator('.toolbar h2')).toHaveText('Orders');
  });

  test('should switch to chat tab', async ({ page }) => {
    await page.locator('.tabs button').nth(2).click();
    await expect(page.locator('.chat-welcome h2')).toHaveText('AI Materials Assistant');
  });

  test('should open material form modal', async ({ page }) => {
    await page.locator('.btn-primary').click();
    await expect(page.locator('.modal')).toBeVisible();
    await expect(page.locator('.modal h3')).toHaveText('New Material');
  });

  test('material form should have stock_quantity field', async ({ page }) => {
    await page.locator('.btn-primary').click();
    const labels = page.locator('.form-grid label');
    const stockLabel = labels.filter({ hasText: 'In Stock' });
    await expect(stockLabel).toBeVisible();
  });

  test('material form should have all required fields', async ({ page }) => {
    await page.locator('.btn-primary').click();
    const nameInput = page.locator('input[placeholder="e.g. Copper Wire 2mm"]');
    await expect(nameInput).toBeVisible();

    const categoryInput = page.locator('input[placeholder="e.g. Metal, Leather, Textile"]');
    await expect(categoryInput).toBeVisible();

    const priceInput = page.locator('input[step="0.01"]');
    await expect(priceInput).toBeVisible();

    const unitSelect = page.locator('select');
    await expect(unitSelect).toBeVisible();

    const minQtyInput = page.locator('input[placeholder=""]');
    await expect(minQtyInput).toBeVisible();

    const stockInput = page.locator('input[type="number"][min="0"]');
    await expect(stockInput).toBeVisible();

    const mfrSelect = page.locator('select').last();
    await expect(mfrSelect).toBeVisible();
  });

  test('should close modal on cancel', async ({ page }) => {
    await page.locator('.btn-primary').click();
    await expect(page.locator('.modal')).toBeVisible();
    await page.locator('.modal .btn').first().click();
    await expect(page.locator('.modal')).not.toBeVisible();
  });

  test('should close modal on overlay click', async ({ page }) => {
    await page.locator('.btn-primary').click();
    await page.locator('.modal-overlay').click({ position: { x: 0, y: 0 } });
    await expect(page.locator('.modal')).not.toBeVisible();
  });

  test('materials cards should have category data attribute', async ({ page }) => {
    // Seed a material via API then check it renders with data-category
    await page.locator('.btn-primary').click();
    await page.locator('input[placeholder="e.g. Copper Wire 2mm"]').fill('Test Copper');
    await page.locator('input[placeholder="e.g. Metal, Leather, Textile"]').fill('Metal');
    await page.locator('input[step="0.01"]').fill('5.00');
    await page.locator('select').first().selectOption('meter');
    await page.locator('input[type="number"][min="1"]').fill('10');
    await page.locator('input[type="number"][min="0"]').fill('100');
    await page.locator('.modal .btn-primary').click();
    // Wait for card to appear
    await page.waitForSelector('.material-card', { timeout: 5000 });
    const card = page.locator('.material-card').first();
    const category = await card.getAttribute('data-category');
    expect(category).toBe('metal');
  });
});
