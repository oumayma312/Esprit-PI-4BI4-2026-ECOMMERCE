import { Component, OnDestroy, OnInit } from '@angular/core';
import { Router } from '@angular/router';

import { AuthService, UserAccount } from '../services/auth.service';
import {
  ProductMatcherCompareResponse,
  ProductMatcherProduct,
  MlService,
} from '../services/ml.service';

interface UploadedImageAsset {
  id: string;
  name: string;
  sizeLabel: string;
  previewUrl: string;
  dataUrl: string;
  createdAtLabel: string;
}

@Component({
  selector: 'app-ai-insights',
  templateUrl: './ai-insights.component.html',
  styleUrls: ['./ai-insights.component.css'],
})
export class AiInsightsComponent implements OnInit, OnDestroy {
  currentUser: UserAccount | null = null;

  uploadMessage = '';

  matcherCompareResult: ProductMatcherCompareResponse | null = null;

  matcherCompareLoading = false;

  uploadedAssets: UploadedImageAsset[] = [];
  selectedUploadedAssetId: string | null = null;
  lastComparedUploadedAssetId: string | null = null;
  lastComparisonSource: 'upload' | null = null;

  constructor(
    private readonly router: Router,
    private readonly auth: AuthService,
    private readonly mlService: MlService,
  ) {}

  ngOnInit(): void {
    this.currentUser = this.auth.current();

    if (!this.currentUser) {
      this.router.navigate(['/login']);
      return;
    }
  }

  ngOnDestroy(): void {
    this.uploadedAssets.forEach((asset) => URL.revokeObjectURL(asset.previewUrl));
  }

  get selectedUploadedAsset(): UploadedImageAsset | null {
    return this.uploadedAssets.find((asset) => asset.id === this.selectedUploadedAssetId) ?? null;
  }

  get comparedUploadedAsset(): UploadedImageAsset | null {
    return this.uploadedAssets.find((asset) => asset.id === this.lastComparedUploadedAssetId) ?? null;
  }

  trackByProduct(_: number, product: ProductMatcherProduct): number {
    return product.productId;
  }

  trackByAsset(_: number, asset: UploadedImageAsset): string {
    return asset.id;
  }

  async onLocalImagesSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement | null;
    const files = Array.from(input?.files ?? []);

    if (!files.length) {
      return;
    }

    const imageFiles = files.filter((file) => file.type.startsWith('image/'));

    if (!imageFiles.length) {
      this.uploadMessage = 'Select at least one valid image to start an analysis.';
      if (input) {
        input.value = '';
      }
      return;
    }

    try {
      const assets = await Promise.all(imageFiles.map((file, index) => this.buildUploadedAsset(file, index)));
      this.uploadedAssets = [...assets, ...this.uploadedAssets];
      this.selectedUploadedAssetId = assets[0]?.id ?? this.selectedUploadedAssetId;
      this.uploadMessage =
        assets.length === 1
          ? '1 image uploaded and ready for analysis.'
          : `${assets.length} images uploaded and ready for analysis.`;

      if (assets[0]) {
        void this.compareUploadedAsset(assets[0]);
      }
    } catch {
      this.uploadMessage = 'The image could not be read locally. Try another file.';
    } finally {
      if (input) {
        input.value = '';
      }
    }
  }

  compareUploadedAsset(asset: UploadedImageAsset): void {
    this.selectedUploadedAssetId = asset.id;
    this.lastComparedUploadedAssetId = asset.id;
    this.lastComparisonSource = 'upload';
    this.matcherCompareLoading = true;
    this.uploadMessage = `Analyzing ${asset.name}...`;

    this.mlService
      .compareMatcherProducts({
        uploadedImageDataUrl: asset.dataUrl,
        uploadedImageName: asset.name,
        limit: 6,
      })
      .subscribe({
        next: (response) => {
          this.matcherCompareResult = response;
          this.matcherCompareLoading = false;
          this.uploadMessage = `Similarity generated for ${asset.name}.`;
        },
        error: () => {
          this.matcherCompareResult = null;
          this.matcherCompareLoading = false;
          this.uploadMessage = 'Similarity could not be calculated for this image.';
        },
      });
  }

  removeUploadedAsset(assetId: string, event?: MouseEvent): void {
    event?.stopPropagation();

    const asset = this.uploadedAssets.find((currentAsset) => currentAsset.id === assetId);
    if (!asset) {
      return;
    }

    const wasSelected = this.selectedUploadedAssetId === assetId;
    const wasCompared = this.lastComparedUploadedAssetId === assetId;

    URL.revokeObjectURL(asset.previewUrl);
    this.uploadedAssets = this.uploadedAssets.filter((currentAsset) => currentAsset.id !== assetId);

    this.selectedUploadedAssetId = this.uploadedAssets[0]?.id ?? null;

    if (!this.uploadedAssets.length) {
      this.matcherCompareResult = null;
      this.lastComparedUploadedAssetId = null;
      this.lastComparisonSource = null;
      this.uploadMessage = 'Image removed. Upload another image to continue.';
      return;
    }

    if (wasCompared) {
      this.matcherCompareResult = null;
      this.lastComparedUploadedAssetId = null;
      this.lastComparisonSource = null;
      void this.compareUploadedAsset(this.uploadedAssets[0]);
      return;
    }

    if (wasSelected) {
      this.uploadMessage = 'Image removed. Another uploaded image is now selected.';
    } else {
      this.uploadMessage = 'Image removed.';
    }
  }

  queryImageUrl(compare: ProductMatcherCompareResponse): string {
    return this.comparedUploadedAsset?.previewUrl ?? compare.queryProduct.imageUrl;
  }

  queryLabel(compare: ProductMatcherCompareResponse): string {
    return this.comparedUploadedAsset?.name ?? compare.queryProduct.productName;
  }

  queryContextLabel(): string {
    return 'Uploaded image';
  }

  private async buildUploadedAsset(file: File, index: number): Promise<UploadedImageAsset> {
    const dataUrl = await this.readFileAsDataUrl(file);
    const previewUrl = URL.createObjectURL(file);
    const createdAt = new Date();

    return {
      id: `${createdAt.getTime()}-${index}-${file.name}`,
      name: file.name,
      sizeLabel: this.formatBytes(file.size),
      previewUrl,
      dataUrl,
      createdAtLabel: createdAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
  }

  private readFileAsDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          resolve(reader.result);
          return;
        }

        reject(new Error('Invalid reader result'));
      };
      reader.onerror = () => reject(reader.error ?? new Error('Unable to read file'));
      reader.readAsDataURL(file);
    });
  }

  private formatBytes(bytes: number): string {
    if (bytes < 1024) {
      return `${bytes} B`;
    }

    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
}
