import { test, expect } from '@playwright/test';

test.describe('Orders Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('.header h1', { timeout: 10000 });
    // Navigate to orders tab
    await page.locator('.tabs button').nth(1).click();
  });

  test('should show orders toolbar', async ({ page }) => {
    await expect(page.locator('.toolbar h2')).toHaveText('Orders');
  });

  test('should have new order button', async ({ page }) => {
    await expect(page.locator('.btn-primary')).toContainText('+ New Order');
  });

  test('should show empty state when no orders', async ({ page }) => {
    await expect(page.locator('.empty')).toBeVisible();
  });

  test('should open new order form', async ({ page }) => {
    await page.locator('.btn-primary').click();
    await expect(page.locator('.modal')).toBeVisible();
    await expect(page.locator('.modal h3')).toHaveText('New Order');
  });

  test('order form should have material select', async ({ page }) => {
    await page.locator('.btn-primary').click();
    const materialSelect = page.locator('.form-grid select');
    await expect(materialSelect).toBeVisible();
  });

  test('order form should have quantity input', async ({ page }) => {
    await page.locator('.btn-primary').click();
    const qtyInput = page.locator('.form-grid input[type="number"]');
    await expect(qtyInput).toBeVisible();
  });

  test('order form should have notes input', async ({ page }) => {
    await page.locator('.btn-primary').click();
    const notesInput = page.locator('.form-grid input[placeholder="Optional notes"]');
    await expect(notesInput).toBeVisible();
  });

  test('should close order form on cancel', async ({ page }) => {
    await page.locator('.btn-primary').click();
    await page.locator('.modal .btn').first().click();
    await expect(page.locator('.modal')).not.toBeVisible();
  });

  test('should switch back to materials tab', async ({ page }) => {
    await page.locator('.tabs button').nth(0).click();
    await expect(page.locator('.toolbar h2')).toHaveText('Inventory');
  });
});
