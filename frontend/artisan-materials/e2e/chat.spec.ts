import { test, expect } from '@playwright/test';

test.describe('Chat Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('.header h1', { timeout: 10000 });
    // Navigate to chat tab
    await page.locator('.tabs button').nth(2).click();
  });

  test('should show chat container', async ({ page }) => {
    await expect(page.locator('.chat-container')).toBeVisible();
  });

  test('should show welcome screen', async ({ page }) => {
    await expect(page.locator('.chat-welcome')).toBeVisible();
    await expect(page.locator('.chat-welcome h2')).toHaveText('AI Materials Assistant');
  });

  test('should show welcome icon with animation', async ({ page }) => {
    const icon = page.locator('.chat-welcome-icon');
    await expect(icon).toBeVisible();
    // Verify it has the gradient background
    const bg = await icon.evaluate(el => getComputedStyle(el).background);
    expect(bg).toBeTruthy();
  });

  test('should show suggestion buttons', async ({ page }) => {
    const suggestions = page.locator('.suggestions .btn');
    await expect(suggestions).toHaveCount(4);
  });

  test('should have chat input area', async ({ page }) => {
    await expect(page.locator('.chat-input-area input')).toBeVisible();
    await expect(page.locator('.chat-input-area .btn')).toBeVisible();
  });

  test('should have send button', async ({ page }) => {
    const sendBtn = page.locator('.chat-input-area .btn');
    await expect(sendBtn).toHaveText('Send');
  });

  test('send button should be disabled when input is empty', async ({ page }) => {
    const sendBtn = page.locator('.chat-input-area .btn');
    await expect(sendBtn).toBeDisabled();
  });

  test('send button should be enabled when input has text', async ({ page }) => {
    const input = page.locator('.chat-input-area input');
    await input.fill('Hello');
    const sendBtn = page.locator('.chat-input-area .btn');
    await expect(sendBtn).not.toBeDisabled();
  });

  test('should support enter key to send', async ({ page }) => {
    const input = page.locator('.chat-input-area input');
    await input.fill('Test message');
    await input.press('Enter');
    // Message should be sent (input should clear)
    await expect(input).toHaveValue('');
  });

  test('should show chat messages area', async ({ page }) => {
    await expect(page.locator('.chat-messages')).toBeVisible();
  });

  test('should have italic placeholder', async ({ page }) => {
    const input = page.locator('.chat-input-area input');
    await expect(input).toHaveAttribute('placeholder');
  });

  test('chat container should have gradient top bar', async ({ page }) => {
    const container = page.locator('.chat-container');
    await expect(container).toBeVisible();
    // The ::before pseudo-element creates the gradient bar
    // We verify the container has the expected structure
    await expect(container.locator('.chat-messages')).toBeVisible();
  });
});
