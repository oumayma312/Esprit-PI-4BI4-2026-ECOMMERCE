import { HighlightPipe } from './highlight.pipe';

describe('HighlightPipe', () => {
  let pipe: HighlightPipe;

  beforeEach(() => {
    pipe = new HighlightPipe();
  });

  it('should create the pipe', () => {
    expect(pipe).toBeTruthy();
  });

  it('should return empty string for empty input', () => {
    expect(pipe.transform('')).toBe('');
  });

  it('should return empty string for null input', () => {
    expect(pipe.transform(null as any)).toBe('');
  });

  it('should escape HTML entities', () => {
    expect(pipe.transform('<script>alert(1)</script>')).toBe('&lt;script&gt;alert(1)&lt;/script&gt;');
  });

  it('should escape ampersands', () => {
    expect(pipe.transform('Tom & Jerry')).toBe('Tom &amp; Jerry');
  });

  it('should convert bold markdown', () => {
    expect(pipe.transform('**bold text**')).toBe('<strong>bold text</strong>');
  });

  it('should convert italic markdown', () => {
    expect(pipe.transform('*italic text*')).toBe('<em>italic text</em>');
  });

  it('should convert inline code', () => {
    expect(pipe.transform('Use `create_order` tool')).toBe('Use <code>create_order</code> tool');
  });

  it('should convert line breaks', () => {
    expect(pipe.transform('line 1\nline 2')).toBe('line 1<br>line 2');
  });

  it('should handle multiple formatting', () => {
    const input = '**Bold** and *italic* and `code`';
    const result = pipe.transform(input);
    expect(result).toContain('<strong>Bold</strong>');
    expect(result).toContain('<em>italic</em>');
    expect(result).toContain('<code>code</code>');
  });

  it('should handle newlines with bold', () => {
    const input = '**Bold line 1**\n*Italic line 2*';
    const result = pipe.transform(input);
    expect(result).toContain('<strong>Bold line 1</strong>');
    expect(result).toContain('<br>');
    expect(result).toContain('<em>Italic line 2</em>');
  });

  it('should not double-escape already escaped HTML', () => {
    const input = '&amp;';
    expect(pipe.transform(input)).toBe('&amp;amp;');
  });

  it('should handle code with special characters', () => {
    expect(pipe.transform('Use `<div>` tag')).toBe('Use <code>&lt;div&gt;</code> tag');
  });
});
