// FIX: To resolve "Exports and export assignments are not permitted in module augmentations",
// this file is converted from a module to a global script file. This is done by
// removing `export {}` and the `declare global` wrapper. Now, `declare module 'mammoth'`
// is a top-level declaration, not an augmentation, and can contain exports.

// These interfaces will be merged into the global scope automatically by TypeScript,
// augmenting Vite's built-in types without conflict.
interface ImportMetaEnv {
  // FIX: Added PROD to the environment type definition to resolve an error when checking for production mode.
  // FIX: Removed `readonly` modifier to match Vite's type declaration for this property and resolve the conflict.
  PROD: boolean;
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_GEMINI_API_KEY?: string;
  readonly VITE_OPENAI_API_KEY?: string;
  readonly VITE_OPENAI_TARGET_URL?: string;
  readonly VITE_OPENAI_ENDPOINT?: string;
  readonly VITE_OPENAI_MODEL?: string;
  readonly VITE_DEEPSEEK_API_KEY?: string;
  readonly VITE_DEEPSEEK_ENDPOINT?: string;
  readonly VITE_DEEPSEEK_MODEL?: string;
  readonly VITE_ALI_API_KEY?: string;
  readonly VITE_ALI_TARGET_URL?: string;
  readonly VITE_ALI_ENDPOINT?: string;
  readonly VITE_ALI_MODEL?: string;
  readonly VITE_DEPOCR_API_KEY?: string;
  readonly VITE_DEPOCR_ENDPOINT?: string;
  readonly VITE_DEPOCR_MODEL?: string;
  readonly VITE_DOUBAO_API_KEY?: string;
  readonly VITE_DOUBAO_ENDPOINT?: string;
  readonly VITE_DOUBAO_MODEL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

interface Window {
  pdfjsLib: any;
}

// Augment the existing NodeJS.ProcessEnv interface for environment variables.
// FIX: Added 'declare' keyword to fix "Top-level declarations in .d.ts files must start with either a 'declare' or 'export' modifier." error.
declare namespace NodeJS {
  interface ProcessEnv {
    readonly API_KEY?: string;
  }
}

// Declare the 'mammoth' module for CommonJS interoperability.
declare module 'mammoth' {
  interface MammothResult {
    value: string;
    messages: any[];
  }

  const mammoth: {
    extractRawText(options: { arrayBuffer: ArrayBuffer }): Promise<MammothResult>;
  };

  // FIX: Using `export =` is the correct syntax for declaring the export of a CJS module.
  // This is now allowed because this file is a global script, not a module.
  export = mammoth;
}