package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// OpenAIError ist die ausgewertete Form des Fehlerstrings.
type OpenAIError struct {
	Method   string          // z.B. "POST"
	URL      string          // z.B. "https://api.openai.com/v1/chat/completions"
	Status   int             // z.B. 429
	Reason   string          // z.B. "Too Many Requests"
	Message  string          // Raw message aus dem Body
	Type     string          // z.B. "tokens"
	Param    *string         // i.d.R. nil
	Code     string          // z.B. "rate_limit_exceeded"
	RateInfo *OpenAIRateInfo // Parse aus message (nur wenn erkannt)
}

func (e *OpenAIError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return e.Method + " " + e.URL + ": " + strconv.Itoa(e.Status) + " " + e.Reason + " - " + e.Message
}

// IsRateLimit meldet true bei 429 oder typischen Rate-Limit-Codes/Strings.
func (e *OpenAIError) IsRateLimit() bool {
	if e == nil {
		return false
	}
	if e.Status == 429 {
		return true
	}
	if e.Code == "rate_limit_exceeded" {
		return true
	}
	msg := strings.ToLower(e.Message)
	return strings.Contains(msg, "rate limit")
}

// IsAuth meldet true bei AuthN/AuthZ-Problemen.
func (e *OpenAIError) IsAuth() bool {
	if e == nil {
		return false
	}
	if e.Status == 401 || e.Status == 403 {
		return true
	}
	switch e.Code {
	case "invalid_api_key", "invalid_api_key_header", "account_deactivated", "organization_deactivated":
		return true
	default:
		return false
	}
}

// IsServerError meldet true bei 5xx.
func (e *OpenAIError) IsServerError() bool {
	if e == nil {
		return false
	}
	return e.Status >= 500 && e.Status <= 599
}

// OpenAIRateInfo enthält feingranulare Rate-Limit-Daten,
// die aus der Message extrahiert werden (falls vorhanden).
type OpenAIRateInfo struct {
	Model      string        // z.B. "gpt-4.1"
	ScopeType  string        // "organization" oder "project" (falls im Text)
	ScopeID    string        // z.B. "org-XXXX"
	Metric     string        // z.B. "tokens per min (TPM)"
	Limit      int           // z.B. 30000
	Used       int           // z.B. 30000
	Requested  int           // z.B. 1741
	RetryAfter time.Duration // z.B. 3.482s
	DocsURL    string        // Rate-limit Doku URL
}

// ParseOpenAIError parst Fehlermeldungen wie:
// POST "https://api.openai.com/v1/chat/completions": 429 Too Many Requests { "message": "...", "type": "...", "param": null, "code": "..." }
func ParseOpenAIJsonError(raw string) (*OpenAIError, error) {
	raw = strings.TrimSpace(raw)

	// 1) Kopf extrahieren
	headRe := regexp.MustCompile(`^(GET|POST|PUT|PATCH|DELETE)\s+"([^"]+)"\s*:\s*(\d{3})\s+([A-Za-z ]+)\s+`)
	m := headRe.FindStringSubmatch(raw)
	e := &OpenAIError{}
	if len(m) == 5 {
		e.Method = m[1]
		e.URL = m[2]
		if s, err := strconv.Atoi(m[3]); err == nil {
			e.Status = s
		}
		e.Reason = strings.TrimSpace(m[4])
	} else {
		return nil, errors.New("unrecognized header format")
	}

	// 2) JSON-Body finden (ab erster '{')
	i := strings.Index(raw, "{")
	if i == -1 {
		return e, nil // kein JSON-Body – geben wir nur Header zurück
	}
	jsonPart := strings.TrimSpace(raw[i:])
	// evtl. escaped \n/\t in echte Whitespace wandeln
	jsonPart = strings.ReplaceAll(jsonPart, `\n`, "\n")
	jsonPart = strings.ReplaceAll(jsonPart, `\t`, "\t")

	// 3) Body unmarshalen – unterstützt beide Varianten:
	//    a) {"error": {...}}
	//    b) {"message": "...", "type": "...", "param": null, "code": "..."}
	var shell struct {
		Error *struct {
			Message string  `json:"message"`
			Type    string  `json:"type"`
			Param   *string `json:"param"`
			Code    string  `json:"code"`
		} `json:"error"`
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Param   *string `json:"param"`
		Code    string  `json:"code"`
	}
	if err := json.Unmarshal([]byte(jsonPart), &shell); err != nil {
		// ggf. abgeschnittene } tolerieren
		if last := strings.LastIndex(jsonPart, "}"); last > 0 {
			if err2 := json.Unmarshal([]byte(jsonPart[:last+1]), &shell); err2 != nil {
				// Body als Text durchreichen
				e.Message = strings.TrimSpace(jsonPart)
				return e, nil
			}
		} else {
			e.Message = strings.TrimSpace(jsonPart)
			return e, nil
		}
	}

	if shell.Error != nil {
		e.Message = shell.Error.Message
		e.Type = shell.Error.Type
		e.Param = shell.Error.Param
		e.Code = shell.Error.Code
	} else {
		e.Message = shell.Message
		e.Type = shell.Type
		e.Param = shell.Param
		e.Code = shell.Code
	}

	// 4) Rate-Limit-Details aus der Message ziehen
	rateRe := regexp.MustCompile(
		`Rate limit reached for ([\w\-.]+) in (organization|project) ([\w-]+) on ([^:]+): Limit (\d+), Used (\d+), Requested (\d+)\. Please try again in ([0-9.]+)s\. Visit (\S+)`,
	)
	rm := rateRe.FindStringSubmatch(e.Message)
	if len(rm) == 10 {
		limit, _ := strconv.Atoi(rm[5])
		used, _ := strconv.Atoi(rm[6])
		req, _ := strconv.Atoi(rm[7])
		sec, _ := strconv.ParseFloat(rm[8], 64)

		e.RateInfo = &OpenAIRateInfo{
			Model:      rm[1],
			ScopeType:  rm[2],
			ScopeID:    rm[3],
			Metric:     strings.TrimSpace(rm[4]),
			Limit:      limit,
			Used:       used,
			Requested:  req,
			RetryAfter: time.Duration(sec * float64(time.Second)),
			DocsURL:    rm[9],
		}
	}

	// Sinnvoller Default-Code bei 429+Rate-Limit
	if e.Code == "" && e.Status == 429 && strings.Contains(strings.ToLower(e.Message), "rate limit") {
		e.Code = "rate_limit_exceeded"
	}
	return e, nil
}

// ParseOpenAIPlainError parst Zeilen wie:
// POST https://api.openai.com/v1/chat/completions: 429 Too Many Requests - Rate limit reached for gpt-4.1 in organization org-... on tokens per min (TPM): Limit 30000, Used 30000, Requested 1895. Please try again in 3.789s. Visit https://platform.openai.com/account/rate-limits to learn more.
func ParseOpenAIPlainError(raw string) (*OpenAIError, error) {
	raw = strings.TrimSpace(raw)

	// Kopf: METHOD URL: STATUS REASON - MESSAGE
	headRe := regexp.MustCompile(`^(GET|POST|PUT|PATCH|DELETE)\s+(\S+):\s+(\d{3})\s+([A-Za-z ]+)\s+-\s+(.*)$`)
	m := headRe.FindStringSubmatch(raw)
	if len(m) != 6 {
		return nil, fmt.Errorf("unrecognized error format")
	}

	status, _ := strconv.Atoi(m[3])
	e := &OpenAIError{
		Method:  m[1],
		URL:     m[2],
		Status:  status,
		Reason:  strings.TrimSpace(m[4]),
		Message: strings.TrimSpace(m[5]),
	}

	// Rate-Limit-Details aus der Message ziehen
	rateRe := regexp.MustCompile(
		`Rate limit reached for ([\w\-.]+) in (organization|project) ([\w-]+) on ([^:]+): Limit (\d+), Used (\d+), Requested (\d+)\. Please try again in ([0-9.]+)s\. Visit (\S+)`,
	)
	rm := rateRe.FindStringSubmatch(e.Message)
	if len(rm) == 10 {
		limit, _ := strconv.Atoi(rm[5])
		used, _ := strconv.Atoi(rm[6])
		req, _ := strconv.Atoi(rm[7])
		sec, _ := strconv.ParseFloat(rm[8], 64)
		if sec > 0 {
			sec = math.Round(sec)
		}

		e.RateInfo = &OpenAIRateInfo{
			Model:      rm[1],
			ScopeType:  rm[2],
			ScopeID:    rm[3],
			Metric:     strings.TrimSpace(rm[4]),
			Limit:      limit,
			Used:       used,
			Requested:  req,
			RetryAfter: time.Duration(sec * float64(time.Second)),
			DocsURL:    rm[9],
		}

		// Type heuristisch aus Metric ableiten
		metricLower := strings.ToLower(e.RateInfo.Metric)
		switch {
		case strings.Contains(metricLower, "token"):
			e.Type = "tokens"
		case strings.Contains(metricLower, "requests"):
			e.Type = "requests"
		default:
			e.Type = "" // unbekannt
		}

		// Sinnvoller Code bei 429
		if e.Status == 429 && strings.Contains(strings.ToLower(e.Message), "rate limit") {
			e.Code = "rate_limit_exceeded"
		}
	}

	return e, nil
}

type innerErr struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    string  `json:"code"`
}
