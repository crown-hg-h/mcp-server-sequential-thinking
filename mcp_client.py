import json
import asyncio
import os
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            api_key=''
        )
        self.conversation_history: List[Dict] = []
        self.current_thought_number = 1
        self.total_thoughts = 1

    async def connect_to_server(self):
        """连接到顺序思考MCP服务器"""
        server_params = StdioServerParameters(
            command='python3',
            args=['sequential_thinking_server.py'],
            env={
                'PATH': os.environ.get('PATH', ''),
                'PYTHONPATH': os.environ.get('PYTHONPATH', '')
            }
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write))

        await self.session.initialize()

    async def process_query(self, query: str) -> str:
        """处理用户查询，使用顺序思考方式"""
        system_prompt = (
            "你是一个使用顺序思考方法的AI助手。"
            "你会将复杂问题分解成多个思考步骤，每个步骤都建立在前面的基础上。"
            "你可以修改之前的想法，开启新的思考分支，并在需要时增加更多的思考步骤。"
            "请记住我们之前的对话内容，并在回答时考虑上下文。"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": query}
        ]

        # 获取所有MCP服务器工具列表信息
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # 初始化思考过程
        while True:
            # 请求GPT生成下一个思考步骤
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=available_tools
            )

            content = response.choices[0]
            if content.finish_reason == "tool_calls":
                tool_call = content.message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 确保thoughtNumber和totalThoughts被正确设置
                if "thoughtNumber" not in tool_args:
                    tool_args["thoughtNumber"] = self.current_thought_number
                if "totalThoughts" not in tool_args:
                    tool_args["totalThoughts"] = max(self.total_thoughts, self.current_thought_number)

                # 执行顺序思考工具
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"\n[思考步骤 {self.current_thought_number}]\n")

                # 解析工具返回的结果
                result_data = json.loads(result.content[0].text)
                self.current_thought_number = result_data.get("thoughtNumber", self.current_thought_number) + 1
                self.total_thoughts = result_data.get("totalThoughts", self.total_thoughts)

                messages.append(content.message.model_dump())
                messages.append({
                    "role": "tool",
                    "content": result.content[0].text,
                    "tool_call_id": tool_call.id,
                })

                # 如果不需要更多思考步骤，生成最终答案
                if not result_data.get("nextThoughtNeeded", True):
                    final_response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                    )
                    answer = final_response.choices[0].message.content
                    
                    # 更新对话历史
                    self.conversation_history.append({"role": "user", "content": query})
                    self.conversation_history.append({"role": "assistant", "content": answer})
                    
                    # 保持对话历史在合理范围内(最近5轮对话)
                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-10:]
                    
                    # 重置思考计数器，为下一次对话做准备
                    self.current_thought_number = 1
                    self.total_thoughts = 1
                    
                    return answer
            else:
                return content.message.content

    async def chat_loop(self):
        """聊天循环"""
        while True:
            try:
                query = input("\n请输入问题 (输入 quit 退出, clear 清空历史): ").strip()

                if query.lower() == 'quit':
                    break
                elif query.lower() == 'clear':
                    self.conversation_history = []
                    self.current_thought_number = 1
                    self.total_thoughts = 1
                    print("\n已清空对话历史")
                    continue

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                import traceback
                traceback.print_exc()

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 