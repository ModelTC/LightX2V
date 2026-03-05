/**
 * Gemini Watermark Remover
 * Removes watermarks from Gemini AI generated images using Reverse Alpha Blending
 */

import { getAssetBasePath } from '../src/utils/assetPath';

// Constants for watermark removal
const ALPHA_THRESHOLD = 0.002;  // Ignore very small alpha values (noise)
const MAX_ALPHA = 0.99;          // Avoid division by near-zero values
const LOGO_VALUE = 255;          // Color value for white watermark

/**
 * Detect watermark configuration based on image size
 */
function detectWatermarkConfig(imageWidth: number, imageHeight: number): { logoSize: number; marginRight: number; marginBottom: number } {
  // Gemini's watermark rules:
  // If both image width and height are greater than 1024, use 96×96 watermark
  // Otherwise, use 48×48 watermark
  if (imageWidth > 1024 && imageHeight > 1024) {
    return {
      logoSize: 96,
      marginRight: 64,
      marginBottom: 64
    };
  } else {
    return {
      logoSize: 48,
      marginRight: 32,
      marginBottom: 32
    };
  }
}

/**
 * Calculate watermark position in image
 */
function calculateWatermarkPosition(
  imageWidth: number,
  imageHeight: number,
  config: { logoSize: number; marginRight: number; marginBottom: number }
): { x: number; y: number; width: number; height: number } {
  const { logoSize, marginRight, marginBottom } = config;
  return {
    x: imageWidth - marginRight - logoSize,
    y: imageHeight - marginBottom - logoSize,
    width: logoSize,
    height: logoSize
  };
}

/**
 * Calculate alpha map from background captured image
 */
function calculateAlphaMap(bgCaptureImageData: ImageData): Float32Array {
  const { width, height, data } = bgCaptureImageData;
  const alphaMap = new Float32Array(width * height);

  // For each pixel, take the maximum value of the three RGB channels and normalize it to [0, 1]
  for (let i = 0; i < alphaMap.length; i++) {
    const idx = i * 4; // RGBA format, 4 bytes per pixel
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];

    // Take the maximum value of the three RGB channels as the brightness value
    const maxChannel = Math.max(r, g, b);

    // Normalize to [0, 1] range
    alphaMap[i] = maxChannel / 255.0;
  }

  return alphaMap;
}

/**
 * Remove watermark using reverse alpha blending
 * Formula: original = (watermarked - α × logo) / (1 - α)
 */
function removeWatermark(
  imageData: ImageData,
  alphaMap: Float32Array,
  position: { x: number; y: number; width: number; height: number }
): void {
  const { x, y, width, height } = position;

  // Process each pixel in the watermark area
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      // Calculate index in original image (RGBA format, 4 bytes per pixel)
      const imgIdx = ((y + row) * imageData.width + (x + col)) * 4;

      // Calculate index in alpha map
      const alphaIdx = row * width + col;

      // Get alpha value
      let alpha = alphaMap[alphaIdx];

      // Skip very small alpha values (noise)
      if (alpha < ALPHA_THRESHOLD) {
        continue;
      }

      // Limit alpha value to avoid division by near-zero
      alpha = Math.min(alpha, MAX_ALPHA);
      const oneMinusAlpha = 1.0 - alpha;

      // Apply reverse alpha blending to each RGB channel
      for (let c = 0; c < 3; c++) {
        const watermarked = imageData.data[imgIdx + c];

        // Reverse alpha blending formula
        const original = (watermarked - alpha * LOGO_VALUE) / oneMinusAlpha;

        // Clip to [0, 255] range
        imageData.data[imgIdx + c] = Math.max(0, Math.min(255, Math.round(original)));
      }

      // Alpha channel remains unchanged
    }
  }
}

/**
 * Load watermark background image
 */
async function loadWatermarkBackground(size: 48 | 96): Promise<ImageData> {
  const img = new Image();
  // 获取资源基础路径，确保在 qiankun/独立部署环境中路径正确
  const basePath = getAssetBasePath();
  const bgPath = size === 48
    ? `${basePath}/assets/bg_48.png`
    : `${basePath}/assets/bg_96.png`;

  return new Promise((resolve, reject) => {
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Failed to get canvas context'));
        return;
      }
      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, size, size);
      resolve(imageData);
    };
    img.onerror = reject;
    img.src = bgPath;
  });
}

/**
 * Remove Gemini watermark from image
 * @param imageInput - Base64 image string or data URL
 * @returns Base64 image string with watermark removed
 */
export const removeGeminiWatermark = async (imageInput: string): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = async () => {
      try {
        // Create canvas to process image
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Failed to get canvas context'));
          return;
        }

        // Draw original image onto canvas
        ctx.drawImage(img, 0, 0);

        // Get image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Detect watermark configuration
        const config = detectWatermarkConfig(canvas.width, canvas.height);
        const position = calculateWatermarkPosition(canvas.width, canvas.height, config);

        // Load and calculate alpha map for watermark size
        const bgImageData = await loadWatermarkBackground(config.logoSize as 48 | 96);
        const alphaMap = calculateAlphaMap(bgImageData);

        // Remove watermark from image data
        removeWatermark(imageData, alphaMap, position);

        // Write processed image data back to canvas
        ctx.putImageData(imageData, 0, 0);

        // Convert canvas to base64 data URL
        const result = canvas.toDataURL('image/png');
        resolve(result);
      } catch (error: any) {
        reject(new Error(`Failed to remove watermark: ${error.message}`));
      }
    };

    img.onerror = () => {
      reject(new Error('Failed to load image'));
    };

    // Handle both data URLs and base64 strings
    const imageSrc = imageInput.includes(',') ? imageInput : `data:image/png;base64,${imageInput}`;
    img.src = imageSrc;
  });
};
