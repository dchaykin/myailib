package openai

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestParseOpenAIError_RateLimit(t *testing.T) {
	raw := `POST "https://api.openai.com/v1/chat/completions": 429 Too Many Requests {
        "message": "Rate limit reached for gpt-4.1 in organization org-YvWUPqaYaDO3IEven3giqHwj on tokens per min (TPM): Limit 30000, Used 30000, Requested 1741. Please try again in 3.482s. Visit https://platform.openai.com/account/rate-limits to learn more.",
        "type": "tokens",
        "param": null,
        "code": "rate_limit_exceeded"
    }`
	e, err := ParseOpenAIJsonError(raw)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if e.Method != "POST" || e.URL == "" {
		t.Fatalf("header parse failed: %+v", e)
	}
	if e.Status != 429 || e.Reason != "Too Many Requests" {
		t.Fatalf("status/reason mismatch: %+v", e)
	}
	if e.Type != "tokens" || e.Code != "rate_limit_exceeded" {
		t.Fatalf("type/code mismatch: %+v", e)
	}
	if e.RateInfo == nil {
		t.Fatalf("expected RateInfo, got nil")
	}
	if e.RateInfo.Model != "gpt-4.1" ||
		e.RateInfo.ScopeType != "organization" ||
		e.RateInfo.ScopeID != "org-YvWUPqaYaDO3IEven3giqHwj" {
		t.Errorf("rate scope mismatch: %+v", e.RateInfo)
	}
	if e.RateInfo.Metric != "tokens per min (TPM)" {
		t.Errorf("metric mismatch: %q", e.RateInfo.Metric)
	}
	if e.RateInfo.Limit != 30000 || e.RateInfo.Used != 30000 || e.RateInfo.Requested != 1741 {
		t.Errorf("limit/used/requested mismatch: %+v", e.RateInfo)
	}
	if e.RateInfo.RetryAfter != 3482*time.Millisecond {
		t.Errorf("retryAfter mismatch: got %v", e.RateInfo.RetryAfter)
	}
	if e.RateInfo.DocsURL == "" {
		t.Errorf("docs URL missing")
	}
	// Classifier
	if !e.IsRateLimit() || e.IsAuth() || e.IsServerError() {
		t.Errorf("classifier mismatch: rateLimit=%v auth=%v server=%v", e.IsRateLimit(), e.IsAuth(), e.IsServerError())
	}
}

func TestParseOpenAIError_WrappedJSON(t *testing.T) {
	raw := `POST https://api.openai.com/v1/chat/completions: 429 Too Many Requests - Rate limit reached for gpt-4.1 in organization org-YvWUPqaYaDO3IEven3giqHwj on tokens per min (TPM): Limit 30000, Used 30000, Requested 1895. Please try again in 3.789s. Visit https://platform.openai.com/account/rate-limits to learn more.`

	e, err := ParseOpenAIPlainError(raw)
	require.NoError(t, err)
	require.EqualValues(t, 429, e.Status)
	require.NotEmpty(t, e.Message)
	require.EqualValues(t, "tokens", e.Type)
	require.EqualValues(t, "rate_limit_exceeded", e.Code)

	require.NotNil(t, e.RateInfo)
	require.True(t, e.IsRateLimit())
	require.EqualValues(t, time.Duration(4000000000), e.RateInfo.RetryAfter)
}

func TestClassifier_IsAuth(t *testing.T) {
	raw := `POST "https://api.openai.com/v1/chat/completions": 401 Unauthorized {
		"error": {
			"message": "Incorrect API key provided",
			"type": "invalid_request_error",
			"param": null,
			"code": "invalid_api_key"
		}
	}`
	e, err := ParseOpenAIJsonError(raw)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if !e.IsAuth() || e.IsRateLimit() || e.IsServerError() {
		t.Errorf("classifier mismatch (auth): auth=%v rate=%v server=%v", e.IsAuth(), e.IsRateLimit(), e.IsServerError())
	}
}

func TestClassifier_IsServerError(t *testing.T) {
	raw := `POST "https://api.openai.com/v1/chat/completions": 502 Bad Gateway {
		"message": "Upstream error"
	}`
	e, err := ParseOpenAIJsonError(raw)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}
	if !e.IsServerError() || e.IsAuth() || e.IsRateLimit() {
		t.Errorf("classifier mismatch (server): server=%v auth=%v rate=%v", e.IsServerError(), e.IsAuth(), e.IsRateLimit())
	}
}
