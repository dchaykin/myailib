package openai

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/stock_analyst/mygolib/log"
)

func NewAiCommunicationService(prompt string) *AiCommunicationService {
	config := config{
		AuthData: map[string]any{
			"apiKey": os.Getenv("OPENAI_API_KEY"),
		},
	}
	return &AiCommunicationService{
		config:      config,
		Prompt:      prompt,
		Model:       openai.ChatModelGPT4_1,
		Temperature: 0.0,
		Costs:       []chatCosts{},
	}
}

type config struct {
	AuthData map[string]any
}

type AiCommunicationService struct {
	config      config
	Model       openai.ChatModel
	Prompt      string
	Costs       []chatCosts
	Temperature float64
}

func (ai *AiCommunicationService) AddCosts(usage openai.CompletionUsage) {
	log.Debug("Prompt Tokens: %d\n", usage.PromptTokens)
	log.Debug("Completion Tokens: %d\n", usage.CompletionTokens)
	log.Debug("Total Tokens: %d\n", usage.TotalTokens)

	promptPrice := 0.005 // USD per 1k tokens
	completionPrice := 0.015
	pt := float64(usage.PromptTokens)
	ct := float64(usage.CompletionTokens)
	cost := (pt/1000.0)*promptPrice + (ct/1000.0)*completionPrice
	log.Debug("Estimated Cost: $%.4f\n", cost)

	ai.Costs = append(ai.Costs, chatCosts{
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		PromptPrice:      promptPrice,
		CompletionPrice:  completionPrice,
		TotalCost:        cost,
	})
}

func (ai AiCommunicationService) TotalCosts() float64 {
	total := 0.0
	for _, cost := range ai.Costs {
		total += cost.TotalCost
	}
	return total
}

/*****************************************************/
/*                    AI COSTS                       */
/*****************************************************/
type chatCosts struct {
	PromptTokens     int64   `json:"promptTokens"`
	CompletionTokens int64   `json:"completionTokens"`
	PromptPrice      float64 `json:"promptPrice"`
	CompletionPrice  float64 `json:"completionPrice"`
	TotalCost        float64 `json:"totalCost"`
}

func (ai AiCommunicationService) apiKey() string {
	if ai.config.AuthData == nil {
		return ""
	}
	apiKey, ok := ai.config.AuthData["apiKey"]
	if !ok || apiKey == nil {
		return ""
	}
	return apiKey.(string)
}

func (ai AiCommunicationService) getFilePart(ctx context.Context, client *openai.Client, fileName string) (*openai.ChatCompletionContentPartUnionParam, error) {
	// Step 1: Lade PDF-Datei
	fileReader, err := os.Open(fileName)
	if err != nil {
		return nil, log.WrapError(err)

	}
	defer fileReader.Close()

	name := func(s []string) string {
		if len(s) > 0 {
			return s[len(s)-1]
		}
		return ""
	}(strings.Split(fileReader.Name(), "/"))

	inputFile := openai.File(fileReader, name, "application/pdf")

	storedFile, err := client.Files.New(ctx, openai.FileNewParams{
		File:    inputFile,
		Purpose: openai.FilePurposeUserData,
	})
	if err != nil {
		return nil, log.WrapError(fmt.Errorf("error uploading file to OpenAI: %s", err.Error()))
	}

	// 2. Create messages
	result := openai.FileContentPart(
		openai.ChatCompletionContentPartFileFileParam{
			FileID: param.NewOpt(storedFile.ID),
		},
	)
	return &result, nil
}

type onGetDocument func(ctx context.Context, client *openai.Client) (*openai.ChatCompletionContentPartUnionParam, error)

func (ai *AiCommunicationService) GenerateContentWithPDF(systemMessage, fileName string) (string, error) {
	return ai.generateJsonContent(systemMessage,
		func(ctx context.Context, client *openai.Client) (*openai.ChatCompletionContentPartUnionParam, error) {
			return ai.getFilePart(ctx, client, fileName)
		},
	)
}

func (ai *AiCommunicationService) GenerateContent(systemMessage string) (string, error) {
	return ai.generateJsonContent(systemMessage, nil)
}

func (ai *AiCommunicationService) generateJsonContent(systemMessage string, f onGetDocument) (string, error) {
	client := openai.NewClient(
		option.WithAPIKey(ai.apiKey()),
	)
	ctx := context.Background()

	messages := []openai.ChatCompletionMessageParamUnion{}

	if systemMessage != "" {
		messages = append(messages, openai.SystemMessage(systemMessage))
	}
	if ai.Prompt != "" {
		messages = append(messages, openai.UserMessage(ai.Prompt))
	}

	if f != nil {
		file, err := f(ctx, &client)
		if err != nil {
			return "", log.WrapError(err)
		}
		messages = append(messages,
			openai.UserMessage(
				[]openai.ChatCompletionContentPartUnionParam{*file},
			),
		)
	}

	var chatCompletion *openai.ChatCompletion
	var err error
	for range 3 {
		chatCompletion, err = client.Chat.Completions.New(ctx,
			openai.ChatCompletionNewParams{
				Messages:    messages,
				Model:       ai.Model,
				Temperature: openai.Float(ai.Temperature),
			})
		if err != nil {
			rawError := err.Error()
			e, err1 := ParseOpenAIJsonError(rawError)
			if err1 != nil {
				e, err1 = ParseOpenAIPlainError(rawError)
			}
			if err1 != nil {
				return "", log.WrapError(err)
			}
			if e.Status == 429 && e.Code == "rate_limit_exceeded" && e.RateInfo != nil {
				// z.B. Backoff/Retry planen:
				time.Sleep(e.RateInfo.RetryAfter + 100*time.Millisecond)
			} else {
				return "", log.WrapError(err)
			}
		} else {
			break
		}
	}
	if err != nil {
		return "", log.WrapError(err)
	}

	finishReason := chatCompletion.Choices[0].FinishReason
	switch finishReason {
	case "stop":
		log.Debug("Chat completion finished successfully.")
	case "length":
		return "", fmt.Errorf("chat completion reached maximum length")
	case "content_filter":
		return "", fmt.Errorf("Chat completion was filtered due to content policy.")
	case "tool_calls":
		return "", fmt.Errorf("Chat completion used tool calls.")
	default:
		return "", fmt.Errorf("Chat completion finished with unknown reason: %s", finishReason)
	}

	// Step 3: Kosten hinzufügen
	ai.AddCosts(chatCompletion.Usage)

	resp := chatCompletion.Choices[0].Message
	content := stripJSONWrapper(resp.Content)
	if content == "" {
		return "", fmt.Errorf("no content returned from OpenAI API")
	}
	log.Debug("Content from OpenAI: %s", content)

	return content, nil
}

func stripJSONWrapper(data string) string {
	msgList := strings.Split(data, "\n")
	for x, xmsg := range msgList {
		xmsg = strings.TrimSpace(xmsg)
		if xmsg == "```json" {
			for y, ymsg := range msgList[x:] {
				ymsg = strings.TrimSpace(ymsg)
				if ymsg == "```" {
					return strings.Join(msgList[x+1:x+y], "\n")
				}
			}
		}
	}
	return data
}

func convertDir(systemMessage, prompt, srcFolder, destFolder string) error {
	aiService := NewAiCommunicationService(prompt)

	entries, err := os.ReadDir(srcFolder)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(destFolder, 0755); err != nil {
		return fmt.Errorf("failed to create destination folder: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		if err := aiService.convertFile(systemMessage, srcFolder, destFolder, entry.Name()); err != nil {
			return err
		}

		log.Info("Converted file: %s", entry.Name())
	}
	return nil
}

func (aiService *AiCommunicationService) convertFile(systemMessage, srcFolder, destFolder, fileName string) error {
	content, err := aiService.GenerateContentWithPDF(systemMessage, srcFolder+"/"+fileName)
	if err != nil {
		return fmt.Errorf("failed to generate content from PDF %s: %w", fileName, err)
	}
	destFilePath := filepath.Join(destFolder, fileName)
	if err := os.WriteFile(destFilePath, []byte(content), 0644); err != nil {
		return fmt.Errorf("failed to write content to file %s: %w", destFilePath, err)
	}
	return nil
}
