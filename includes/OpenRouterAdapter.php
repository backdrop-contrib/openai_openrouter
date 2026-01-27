<?php

/**
 * @file
 * OpenRouter adapter for accessing 200+ AI models.
 *
 * Image generation note:
 * - Use /api/v1/chat/completions with "modalities": ["image","text"]
 * - Responses usually contain a data URL at choices[0].message.images[0].image_url.url
 * - We DO NOT stream for images; we need the final JSON to extract the data URL.
 */

// Prefer Composer Manager's autoloader when available; fall back to module vendor.
$__openai_sdk_source =& backdrop_static('openai_sdk_source');
if (module_exists('composer_manager')) {
  if (function_exists('composer_manager_register_autoloader')) {
    composer_manager_register_autoloader();
  }
  $__openai_sdk_source = 'composer_manager';
}
else {
  $autoload = BACKDROP_ROOT . '/' . backdrop_get_path('module', 'openai') . '/vendor/autoload.php';
  if (file_exists($autoload)) {
    require_once $autoload;
    $__openai_sdk_source = 'module_vendor';
  }
  else {
    $__openai_sdk_source = $__openai_sdk_source ?: 'unknown';
  }
}

use OpenAI\Client as OpenAIClient;
use OpenAI\Exceptions\TransporterException;

class OpenRouterAdapter {

  /** @var OpenAIClient */
  protected $client;

  /** @var OpenAIApi|null Reference to parent API for helper methods */
  protected $api;

  /** @var string The actual OpenRouter API key */
  protected $realApiKey;

  public function __construct($apiKey, ?OpenAIApi $api = NULL) {
    $apiKey = trim($apiKey);
    $this->realApiKey = $apiKey;

    // Dummy OpenAI-looking key for SDK validation, override with headers.
    $dummyKey = 'sk-1234567890abcdef1234567890abcdef1234567890abcd';

    $this->client = \OpenAI::factory()
      ->withApiKey($dummyKey)
      ->withBaseUri('https://openrouter.ai/api/v1')
      ->withHttpHeader('Authorization', 'Bearer ' . $apiKey)
      ->withHttpHeader('HTTP-Referer', url('<front>', ['absolute' => TRUE]))
      ->withHttpHeader('X-Title', config_get('system.core', 'site_name') ?: 'Backdrop CMS')
      ->make();

    $this->api = $api;
  }

  /** ------------------------ Models ------------------------ */

  public function getModels(): array {
    $models = [];
    try {
      $model_data = $this->fetchModelData();

      // Build array with provider info for sorting
      $models_with_provider = [];
      foreach ($model_data as $model) {
        $id = $model['id'] ?? '';
        $name = $model['name'] ?? $id;
        if ($id) {
          // Extract provider from ID (e.g., "google/gemini" -> "Google")
          $provider = '';
          if (strpos($id, '/') !== FALSE) {
            list($provider) = explode('/', $id, 2);
            $provider = ucfirst($provider);
          }
          $models_with_provider[$id] = [
            'name' => $name,
            'provider' => $provider,
          ];
        }
      }

      // Sort by provider first, then by name
      uasort($models_with_provider, function($a, $b) {
        $provider_cmp = strcmp($a['provider'], $b['provider']);
        if ($provider_cmp !== 0) {
          return $provider_cmp;
        }
        return strcmp($a['name'], $b['name']);
      });

      // Convert back to simple id => name array
      foreach ($models_with_provider as $id => $info) {
        $models[$id] = $info['name'];
      }

      // Filter by enabled models if configured
      $enabled_models = config_get('openai_openrouter.settings', 'enabled_models');
      // If enabled_models exists (even if empty array), filter by it
      // This means: once the config exists, admin must explicitly enable models
      if (isset($enabled_models) && is_array($enabled_models)) {
        if (empty($enabled_models)) {
          // No models enabled = return empty array
          return [];
        }
        // Filter but preserve the sorted order from $models
        $filtered = [];
        foreach ($models as $id => $name) {
          if (in_array($id, $enabled_models)) {
            $filtered[$id] = $name;
          }
        }
        return $filtered;
      }
    } catch (\Exception $e) {
      watchdog('openai', 'Failed to fetch OpenRouter models: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
    }
    return $models;
  }

  protected function fetchModelData(): array {
    $cache_key = 'openrouter_model_data';
    $cached = cache_get($cache_key);
    if ($cached && !empty($cached->data)) return $cached->data;

    try {
      $ch = curl_init('https://openrouter.ai/api/v1/models');
      curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => TRUE,
        CURLOPT_HTTPHEADER => [
          'Authorization: Bearer ' . $this->realApiKey,
          'HTTP-Referer: ' . url('<front>', ['absolute' => TRUE]),
          'X-Title: ' . (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
        ],
      ]);
      $response = curl_exec($ch);
      $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
      curl_close($ch);

      if ($http_code == 200) {
        $data = json_decode($response, TRUE);
        $model_data = $data['data'] ?? [];
        cache_set($cache_key, $model_data, 'cache', time() + 3600);
        return $model_data;
      }
      throw new \Exception('HTTP ' . $http_code . ': ' . substr((string)$response, 0, 200));
    } catch (\Exception $e) {
      watchdog('openai', 'Failed to fetch OpenRouter model data: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return [];
    }
  }

  /**
   * Get models for a given capability.
   *
   * OpenRouter's own capability metadata has been found to be unreliable:
   * some models misreport or omit their supported modalities (text, image,
   * embeddings, etc.). To avoid breaking behaviour based on that metadata,
   * this adapter intentionally ignores the requested capability and returns
   * the full model list from getModels().
   *
   * Site administrators who need more precise capability information can
   * override or adjust model capabilities via
   * hook_openai_model_capabilities_alter(), which is applied to the data
   * returned by this adapter.
   *
   * @param string $capability
   *   The capability to nominally filter by (e.g., 'text', 'image',
   *   'embeddings'). Currently used only as a hint; no filtering is applied
   *   due to the limitations described above.
   *
   * @return array
   *   The list of models, unfiltered by capability.
   */
  public function getModelsByCapability($capability): array {
    return $this->getModels();
  }

  protected function fetchEmbeddingModels(): array {
    // OpenRouter doesn't have a separate embeddings endpoint
    // Embedding models are included in the main /models response
    // So we just return empty array here and rely on the main model list
    return [];
  }

  public function getModerationModels(): array {
    return []; // not exposed via OpenRouter
  }

  /**
   * Get chat/text models.
   */
  public function getChatModels(): array {
    return $this->getModelsByCapability('text');
  }

  /**
   * Get image generation models.
   */
  public function getImageModels(): array {
    return $this->getModelsByCapability('image');
  }

  /**
   * Get vision models (image input).
   */
  public function getVisionModels(): array {
    return $this->getModelsByCapability('vision');
  }

  /** ------------------------ Text / Chat ------------------------ */

  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    try {
      $payload = [
        'model' => $model,
        'prompt' => trim($prompt),
        'temperature' => (float)$temperature,
        'max_tokens' => max(1, min((int)$max_tokens ?: 1024, 8192)),
      ];

      // Streaming paths removed: always use the non-streaming HTTP response

      $ch = curl_init('https://openrouter.ai/api/v1/completions');
      curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => TRUE,
        CURLOPT_POST => TRUE,
        CURLOPT_POSTFIELDS => json_encode($payload),
        CURLOPT_HTTPHEADER => [
          'Content-Type: application/json',
          'Authorization: Bearer ' . $this->realApiKey,
          'HTTP-Referer: ' . url('<front>', ['absolute' => TRUE]),
          'X-Title' => (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
        ],
      ]);
      $response = curl_exec($ch);
      $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
      curl_close($ch);

      if ($http_code == 200) {
        $result = json_decode($response, TRUE);
        return trim($result['choices'][0]['text'] ?? '');
      }
      throw new \Exception('HTTP ' . $http_code . ': ' . substr((string)$response, 0, 500));
    } catch (TransporterException | \Exception $e) {
      watchdog('openai', 'OpenRouter completions error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return '';
    }
  }

  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE) {
    try {
      $payload = [
        'model' => $model,
        'messages' => $messages,
        'temperature' => (float)$temperature,
        'max_tokens' => max(1, min((int)$max_tokens ?: 1024, 8192)),
      ];

      // Streaming paths removed: always use the non-streaming HTTP response

      $ch = curl_init('https://openrouter.ai/api/v1/chat/completions');
      curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => TRUE,
        CURLOPT_POST => TRUE,
        CURLOPT_POSTFIELDS => json_encode($payload),
        CURLOPT_HTTPHEADER => [
          'Content-Type: application/json',
          'Authorization: Bearer ' . $this->realApiKey,
          'HTTP-Referer: ' . url('<front>', ['absolute' => TRUE]),
          'X-Title' => (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
        ],
      ]);
      $response = curl_exec($ch);
      $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
      curl_close($ch);

      if ($http_code == 200) {
        $result = json_decode($response, TRUE);
        return trim($result['choices'][0]['message']['content'] ?? '');
      }

      // If we got a 400 error about system instructions not being enabled,
      // retry without system messages
      if ($http_code == 400) {
        $error_data = json_decode($response, TRUE);

        // Check if the error is about system/developer instructions
        // The error might be nested in metadata.raw
        $should_retry = false;

        // Check top-level error message
        if (isset($error_data['error']['message'])) {
          $error_msg = $error_data['error']['message'];
          if (stripos($error_msg, 'Developer instruction is not enabled') !== FALSE) {
            $should_retry = true;
          }
        }

        // Check nested metadata.raw if present
        if (!$should_retry && isset($error_data['error']['metadata']['raw'])) {
          $raw_error = $error_data['error']['metadata']['raw'];
          if (stripos($raw_error, 'Developer instruction is not enabled') !== FALSE) {
            $should_retry = true;
          }
        }

        if ($should_retry) {
          // Strip out system messages and retry
          $messages_no_system = [];
          foreach ($messages as $message) {
            if (isset($message['role']) && $message['role'] === 'system') {
              // Convert system message to user message
              $messages_no_system[] = [
                'role' => 'user',
                'content' => '[Instructions]: ' . $message['content'],
              ];
            } else {
              $messages_no_system[] = $message;
            }
          }

          // Retry with modified messages
          $payload['messages'] = $messages_no_system;

          $ch = curl_init('https://openrouter.ai/api/v1/chat/completions');
          curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => TRUE,
            CURLOPT_POST => TRUE,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
              'Content-Type: application/json',
              'Authorization: Bearer ' . $this->realApiKey,
              'HTTP-Referer: ' . url('<front>', ['absolute' => TRUE]),
              'X-Title' => (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
            ],
          ]);
          $response = curl_exec($ch);
          $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
          curl_close($ch);

          if ($http_code == 200) {
            $result = json_decode($response, TRUE);
            return trim($result['choices'][0]['message']['content'] ?? '');
          }
        }
      }

      throw new \Exception('HTTP ' . $http_code . ': ' . substr((string)$response, 0, 500));
    } catch (TransporterException | \Exception $e) {
      watchdog('openai', 'OpenRouter chat error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return '';
    }
  }

  /** ------------------------ Image Generation ------------------------ */

  protected function mapSizeToAspectRatio(string $size): ?string {
    switch ($size) {
      case '1024x1024': return '1:1';
      case '1792x1024': return '16:9';
      case '1024x1792': return '9:16';
      case '1248x832':  return '3:2';
      case '832x1248':  return '2:3';
      default: return NULL;
    }
  }

  /**
   * Generate an image via OpenRouter chat/completions with modalities.
   * Returns: ['data' => [ ['url' => 'data:image/png;base64,...'] ]]  or
   *          ['data' => [ ['b64_json' => '...'] ]]
   */
  public function images(
    string $model,
    string $prompt,
    string $size = '1024x1024',
    string $response_format = 'url',
    string $quality = 'standard',
    string $style = 'natural',
    ?string $output_format = NULL
  ) {
    // IMPORTANT: do not stream for image gen; we need the final JSON.
    // Build messages with an optional system prompt. Many OpenRouter-like
    // models will return text unless explicitly instructed to produce an
    // image. Allow a site-configured system prompt (openai.settings) to be
    // prepended; otherwise use a conservative default instruction that asks
    // the model to generate an image response (data URL or image object).
    $system_prompt = config_get('openai.settings', 'images_system_prompt') ?: config_get('openai.settings', 'image_system_prompt') ?: NULL;
    if (empty($system_prompt)) {
      // Conservative default that nudges model toward producing image output
      // rather than plain text; site admins can override via config.
      $system_prompt = "You are an image generation model. When responding, produce an image and include either a data URL (data:image/...) or an image object in the response. Do not return only text.";
    }

    $messages = [];
    if (!empty($system_prompt)) {
      $messages[] = ['role' => 'system', 'content' => $system_prompt];
    }
    $messages[] = ['role' => 'user', 'content' => $prompt];

    $payload = [
      'model' => $model,
      'messages' => $messages,
      'modalities' => ['image', 'text'],
      'max_tokens' => 2048,
    ];
    if ($aspect = $this->mapSizeToAspectRatio($size)) {
      $payload['image_config'] = ['aspect_ratio' => $aspect];
    }

    try {
      $ch = curl_init('https://openrouter.ai/api/v1/chat/completions');
      curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => TRUE,
        CURLOPT_POST => TRUE,
        CURLOPT_POSTFIELDS => json_encode($payload),
        CURLOPT_HTTPHEADER => [
          'Content-Type: application/json',
          'Authorization: Bearer ' . $this->realApiKey,
          'HTTP-Referer' => url('<front>', ['absolute' => TRUE]),
          'X-Title' => (config_get('system.core', 'site_name') ?: 'Backdrop CMS'),
        ],
      ]);

      $response  = curl_exec($ch);
      $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
      curl_close($ch);

      if ($http_code !== 200 || empty($response)) {
        throw new \Exception('HTTP ' . $http_code . ': ' . substr((string)$response, 0, 500));
      }

      $result = json_decode($response, TRUE);
      // Try documented field first:
      $message = $result['choices'][0]['message'] ?? [];
      $dataUrl = $message['images'][0]['image_url']['url'] ?? NULL;

      if (!$dataUrl) {
        // Some providers put image objects in content or other keys. Search broadly.
        $found = $this->search_for_image_in_response($result);
        if (!empty($found['url'])) {
          $dataUrl = $found['url'];
        } elseif (!empty($found['b64_json'])) {
          return ['data' => [ ['b64_json' => $found['b64_json']] ]];
        }
      }

      if (!$dataUrl) {
        return ['data' => []];
      }

      // If it is a data URL, optionally return raw base64.
      if (is_string($dataUrl) && strpos($dataUrl, 'data:image/') === 0) {
        $comma = strpos($dataUrl, ',');
        $b64 = ($comma !== FALSE) ? substr($dataUrl, $comma + 1) : '';
        if ($response_format === 'b64_json') {
          return ['data' => [ ['b64_json' => $b64] ]];
        }
        return ['data' => [ ['url' => $dataUrl] ]];
      }

      // If it happens to be a regular https URL, return as-is.
      if (filter_var($dataUrl, FILTER_VALIDATE_URL)) {
        return ['data' => [ ['url' => $dataUrl] ]];
      }

      return ['data' => []];
    } catch (\Exception $e) {
      watchdog('openai_openrouter', 'OpenRouter images() failed: @err', ['@err' => $e->getMessage()], WATCHDOG_WARNING);
      return ['data' => []];
    }
  }

  /**
   * Recursively search response for image URL or base64.
   * Returns ['url' => ..., 'b64_json' => ...] (one or both).
   */
  protected function search_for_image_in_response($data) {
    $out = ['url' => NULL, 'b64_json' => NULL];

    if (is_string($data)) {
      if (filter_var(trim($data), FILTER_VALIDATE_URL)) { $out['url'] = trim($data); return $out; }
      if (preg_match('/https?:\/\/[^\s)\"]+\.(png|jpg|jpeg|webp|gif)/i', $data, $m)) { $out['url'] = $m[0]; return $out; }
      if (strpos($data, 'data:image/') === 0) { $out['url'] = $data; return $out; }
      if (preg_match('/data:image\/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+\/=\n\r]+)/i', $data, $m2)) {
        $out['b64_json'] = $m2[2]; return $out;
      }
      return $out;
    }

    if (is_array($data)) {
      $container_keys = ['images','outputs','output','artifacts','data','result','response','choices','content'];
      foreach ($container_keys as $ck) {
        if (isset($data[$ck])) {
          $found = $this->search_for_image_in_response($data[$ck]);
          if ($found['url'] || $found['b64_json']) return $found;
        }
      }
      foreach ($data as $k => $v) {
        $lk = strtolower((string)$k);
        if (is_string($v)) {
          if (strpos($lk, 'b64') !== FALSE || strpos($lk, 'base64') !== FALSE) {
            if (preg_match('/([A-Za-z0-9+\/=\n\r]{100,})/', $v, $m3)) { $out['b64_json'] = $m3[1]; return $out; }
          }
          if (strpos($lk, 'url') !== FALSE || strpos($lk, 'link') !== FALSE) {
            if (filter_var(trim($v), FILTER_VALIDATE_URL) || strpos($v, 'data:image/') === 0) { $out['url'] = trim($v); return $out; }
          }
        }
        if (is_array($v) || is_string($v)) {
          $found = $this->search_for_image_in_response($v);
          if ($found['url'] || $found['b64_json']) return $found;
        }
      }
    }

    return $out;
  }

  /** ------------------------ Audio / Embeddings / Moderation ------------------------ */

  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    try {
      return $this->client->audio()->speech([
        'model' => $model,
        'voice' => $voice,
        'input' => $input,
        'response_format' => $response_format,
      ]);
    } catch (TransporterException | \Exception $e) {
      watchdog('openai', 'OpenRouter TTS error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return '';
    }
  }

  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    if (!in_array($task, ['transcribe', 'translate'], TRUE)) {
      throw new \InvalidArgumentException('Task must be transcribe or translate.');
    }
    try {
      $response = $this->client->audio()->$task([
        'model' => $model,
        'file' => fopen($file, 'r'),
        'temperature' => (float)$temperature,
        'response_format' => $response_format,
      ])->toArray();

      return $response['text'] ?? '';
    } catch (TransporterException | \Exception $e) {
      watchdog('openai', 'OpenRouter STT error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return '';
    }
  }

  public function moderation(string $input, string $model = 'omni-moderation-latest'): array {
    try {
      return $this->client->moderations()->create([
        'model' => $model,
        'input' => trim($input),
      ])->toArray();
    } catch (TransporterException | \Exception $e) {
      watchdog('openai', 'OpenRouter moderation error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      return [];
    }
  }

  public function embedding(string $input, string $model, bool $log = TRUE): array {
    $start_time = microtime(TRUE);
    try {
      $response = $this->client->embeddings()->create([
        'model' => $model,
        'input' => $input,
      ])->toArray();
      $result = $response['data'][0]['embedding'] ?? [];
      if (isset($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], $response, TRUE, $duration, NULL, !$log);
      }
      return $result;
    } catch (TransporterException | \Exception $e) {
      if (isset($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], NULL, FALSE, $duration, $e->getMessage(), !$log);
      }
      if ($log) {
        $error_msg = $e->getMessage();
        // Suppress log if it's a "does not support embeddings" or similar during probing.
        if (strpos($error_msg, 'does not support embeddings') === FALSE && strpos($error_msg, 'not found') === FALSE) {
          watchdog('openai', 'OpenRouter embedding error: @error', ['@error' => $error_msg], WATCHDOG_ERROR);
        }
      }
      return [];
    }
  }

  public function getEmbeddingModels(): array {
    return $this->getModelsByCapability('embeddings');
  }
}
