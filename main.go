package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
	"readline"
	//"github.com/chzyer/readline"
)

// 配置参数
var (
	apiKey       = flag.String("key", os.Getenv("ABL_API_KEY"), "API密钥(可使用变量ABL_API_KEY)")
	defaultModel = flag.String("model", "qwen-plus", "默认模型名称")
	apiEndpoint  = flag.String("api", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "百炼API")
	timeoutSec   = flag.Int("timeout", 300, "请求超时时间（秒）")
	historyFile  = flag.String("history", "", "历史记录文件路径")
	command      = flag.String("c", "", "直接执行单条命令后退出")
	enableStream = flag.Bool("stream", false, "在 -c 模式下启用流式输出")
	enableDebug  = flag.Bool("debug", false, "初始调试模式状态")
)

// 数据结构
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type StreamRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type StreamResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage struct {
		PromptTokens    int `json:"prompt_tokens,omitempty"`
		CompletionTokens int `json:"completion_tokens,omitempty"`
		TotalTokens     int `json:"total_tokens,omitempty"`
	} `json:"usage,omitempty"`
}

// 对话状态
type ChatState struct {
	Model         string
	History       []Message
	CmdHistory    []string
	Client        *http.Client
	Debug         bool
	LastRequestID string
	isSingleCmd   bool
}

func main() {
	flag.Parse()
	validateConfig()

	client := &http.Client{
		Timeout: time.Duration(*timeoutSec) * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			IdleConnTimeout:     30 * time.Second,
			DisableCompression:  false,
		},
	}

	chatState := &ChatState{
		Model:       *defaultModel,
		History:     []Message{{Role: "system", Content: "You are a helpful assistant."}},
		CmdHistory:  []string{},
		Client:      client,
		Debug:       *enableDebug,
		isSingleCmd: *command != "",
	}

	if *command != "" {
		if err := executeSingleCommand(chatState, *command); err != nil {
			fmt.Fprintln(os.Stderr, "错误:", err)
			os.Exit(1)
		}
		return
	}

	startInteractiveSession(chatState)
}

func validateConfig() {
	if *apiKey == "" {
		fmt.Fprintln(os.Stderr, "错误：必须提供API密钥")
		flag.Usage()
		os.Exit(1)
	}
}

func executeSingleCommand(state *ChatState, cmd string) error {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return errors.New("空命令")
	}

	state.CmdHistory = append(state.CmdHistory, cmd)

	if handleCommand(cmd, state) {
		return nil
	}

	state.History = append(state.History, Message{Role: "user", Content: cmd})
	_, err := processAIResponse(state, *enableStream)
	return err
}

func startInteractiveSession(state *ChatState) {
	rl, err := readline.NewEx(&readline.Config{
		Prompt:          "> ",
		HistoryFile:     getHistoryFilePath(),
		AutoComplete:    getCompleter(),
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "初始化命令行失败: %v\n", err)
		os.Exit(1)
	}
	defer rl.Close()

	printWelcomeMessage(state)

	for {
		input, err := rl.Readline()
		if err != nil {
			if err == readline.ErrInterrupt {
				if len(input) == 0 {
					break
				}
				continue
			} else if err == io.EOF {
				break
			}
			fmt.Fprintf(os.Stderr, "读取输入错误: %v\n", err)
			continue
		}

		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}

		state.CmdHistory = append(state.CmdHistory, input)

		if handleCommand(input, state) {
			continue
		}

		state.History = append(state.History, Message{Role: "user", Content: input})
		if _, err := processAIResponse(state, true); err != nil {
			fmt.Fprintf(os.Stderr, "\n错误: %v\n", err)
		}
		fmt.Println()
	}
}

func getHistoryFilePath() string {
	if *historyFile != "" {
		return *historyFile
	}
	return os.TempDir() + "/abls_history.txt"
}

func getCompleter() *readline.PrefixCompleter {
	return readline.NewPrefixCompleter(
		readline.PcItem("/model",
			readline.PcItem("qwen-plus"),
			readline.PcItem("qwen-max"),
			readline.PcItem("qwen-turbo"),
			readline.PcItem("deepseek-r1"),
			readline.PcItem("deepseek-v3"),
		),
		readline.PcItem("/debug"),
		readline.PcItem("/reset"),
		readline.PcItem("/help"),
		readline.PcItem("/history"),
		readline.PcItem("exit"),
	)
}

func handleCommand(input string, state *ChatState) bool {
	switch {
	case input == "exit" || input == "/quit":
		os.Exit(0)
	case input == "/reset":
		resetConversation(state)
		return true
	case strings.HasPrefix(input, "/model"):
		handleModelSwitch(input, state)
		return true
	case input == "/debug":
		toggleDebugMode(state)
		return true
	case input == "/help":
		printHelp()
		return true
	case input == "/history":
		showCommandHistory(state)
		return true
	}
	return false
}

func resetConversation(state *ChatState) {
	state.History = []Message{{Role: "system", Content: "You are a helpful assistant."}}
	state.LastRequestID = ""
	fmt.Println("对话历史已重置")
}

func handleModelSwitch(input string, state *ChatState) {
	parts := strings.Split(input, " ")
	if len(parts) < 2 {
		fmt.Printf("当前模型: %s\n可用模型: qwen-plus, qwen-max, qwen-turbo, deepseek-r1, deepseek-v3\n", state.Model)
		return
	}

	newModel := parts[1]
	switch newModel {
	case "qwen-plus", "qwen-max", "qwen-turbo", "deepseek-r1", "deepseek-v3":
		state.Model = newModel
		fmt.Printf("已切换模型为: %s\n", state.Model)
	default:
		fmt.Println("错误：不支持的模型")
	}
}

func toggleDebugMode(state *ChatState) {
	state.Debug = !state.Debug
	fmt.Printf("调试模式 %v\n", state.Debug)
}

func showCommandHistory(state *ChatState) {
	if len(state.CmdHistory) == 0 {
		fmt.Println("暂无历史记录")
		return
	}

	fmt.Println("命令历史:")
	for i, cmd := range state.CmdHistory {
		fmt.Printf("%4d: %s\n", i+1, cmd)
	}
}

func processAIResponse(state *ChatState, streamOutput bool) (string, error) {
	startTime := time.Now()
	
	if streamOutput && !state.isSingleCmd {
		fmt.Printf("AI(%s): ", state.Model)
	}

	aiReply, requestID, err := streamChatCompletion(state, streamOutput)
	if err != nil {
		return "", err
	}

	state.LastRequestID = requestID
	state.History = append(state.History, Message{
		Role:    "assistant",
		Content: aiReply,
	})

	if state.isSingleCmd {
		if !streamOutput {
			fmt.Println(aiReply)
		}
	} else if !streamOutput {
		fmt.Println(aiReply)
	}

	if state.Debug {
		printDebugInfo(startTime, state)
	}

	return aiReply, nil
}

func streamChatCompletion(state *ChatState, streamOutput bool) (string, string, error) {
	payload := StreamRequest{
		Model:    state.Model,
		Messages: state.History,
		Stream:   true,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", "", fmt.Errorf("JSON编码失败: %w", err)
	}

	if state.Debug {
		fmt.Printf("\n[DEBUG] 请求体: %s\n", jsonData)
	}

	req, err := http.NewRequest("POST", *apiEndpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", "", fmt.Errorf("创建请求失败: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+*apiKey)

	resp, err := state.Client.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("请求发送失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", "", fmt.Errorf("API错误 %d: %s", resp.StatusCode, string(body))
	}

	return processStreamResponse(resp.Body, state.Debug, streamOutput)
}

func processStreamResponse(body io.Reader, debug, streamOutput bool) (string, string, error) {
	reader := bufio.NewReader(body)
	var (
		fullResponse strings.Builder
		requestID    string
	)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return "", "", fmt.Errorf("读取流失败: %w", err)
		}

		if len(line) < 6 || !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}

		if bytes.Equal(line, []byte("data: [DONE]\n")) {
			break
		}

		var chunk StreamResponse
		if err := json.Unmarshal(line[6:], &chunk); err != nil {
			return "", "", fmt.Errorf("解析JSON失败: %w", err)
		}

		if debug {
			fmt.Printf("\n[DEBUG] 收到数据块: %+v\n", chunk)
		}

		if requestID == "" && chunk.ID != "" {
			requestID = chunk.ID
		}

		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			if content != "" {
				if streamOutput {
					fmt.Print(content)
				}
				fullResponse.WriteString(content)
			}

			if chunk.Choices[0].FinishReason == "stop" {
				break
			}
		}
	}

	if fullResponse.Len() == 0 {
		return "", "", errors.New("未收到有效回复内容")
	}

	return fullResponse.String(), requestID, nil
}

func printDebugInfo(startTime time.Time, state *ChatState) {
	fmt.Printf("\n[DEBUG] 本次请求耗时: %.2fs\n", time.Since(startTime).Seconds())
	fmt.Printf("[DEBUG] 请求ID: %s\n", state.LastRequestID)
	fmt.Printf("[DEBUG] 当前历史消息数: %d\n", len(state.History))
	fmt.Printf("[DEBUG] 最后一条历史消息: %+v\n", state.History[len(state.History)-1])
}

func printWelcomeMessage(state *ChatState) {
	fmt.Printf(`
阿里云百炼对话客户端
----------------------------------
当前配置:
  模型: %s
  调试模式: %v
  历史记录文件: %s
----------------------------------
命令说明:
  /help        显示帮助
  /reset       重置对话
  /model <模型名> 切换模型
  /debug       切换调试模式
  /history     查看命令历史
  exit         退出程序
----------------------------------
`, state.Model, state.Debug, getHistoryFilePath())
}

func printHelp() {
	fmt.Println(`
交互命令:
  /help        显示本帮助
  /reset       清除对话历史
  /model       显示/切换模型
  /debug       切换调试信息
  /history     查看命令历史
  exit         退出程序

单命令模式选项:
  -c string    执行单条命令后退出
  --stream     在单命令模式下启用流式输出

使用示例:
  # 单命令普通模式
  ./abls -c "你好"
  
  # 单命令流式模式
  ./abls -c "你好" --stream
  
  # 交互模式
  ./abls`)
}