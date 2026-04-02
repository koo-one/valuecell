export type MemoryItem = {
  id: number;
  content: string;
};

export type ModelProvider = {
  provider: string;
};

export type ProviderModelInfo = {
  model_id: string;
  model_name: string;
};

export type ProviderDetail = {
  api_key?: string | null;
  api_key_url?: string | null;
  base_url?: string | null;
  is_default: boolean;
  default_model_id?: string | null;
  auth_type?: string | null;
  oauth_authenticated: boolean;
  oauth_expires_at?: number | null;
  oauth_account_id?: string | null;
  models: ProviderModelInfo[];
};

// --- Model availability check ---
export type CheckModelRequest = {
  provider?: string;
  model_id?: string;
  api_key?: string;
};

export type CheckModelResult = {
  ok: boolean;
  provider: string;
  model_id: string;
  status?: string;
  error?: string;
};
