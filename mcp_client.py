import json
import asyncio
import os
import httpx
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from fastmcp import FastMCP

# 初始化 FastMCP 服务器
app = FastMCP('web-search')

@app.tool()
async def web_search(query: str) -> str:
    """
    搜索互联网内容

    Args:
        query: 要搜索的内容

    Returns:
        搜索结果的总结
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'https://open.bigmodel.cn/api/paas/v4/tools',
            headers={'Authorization': '2ec0f0dceab6449ebd81e38795b97320.66BtB5l6HE4M7Yq7'},
            json={
                'tool': 'web-search-pro',
                'messages': [
                    {'role': 'user', 'content': query}
                ],
                'stream': False
            }
        )

        res_data = []
        for choice in response.json()['choices']:
            for message in choice['message']['tool_calls']:
                search_results = message.get('search_result')
                if not search_results:
                    continue
                for result in search_results:
                    res_data.append(result['content'])

        return '\n\n\n'.join(res_data)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key='3bc812cb-fc28-4f79-8d16-5c9114796bc5'
        )
        self.current_thought_number = 1
        self.total_thoughts = 1
        self.conversation_history = []  # 新增：用于保存整个对话过程

    async def connect_to_server(self):
        """连接到顺序思考MCP服务器"""
        server_params = StdioServerParameters(
            command='node',
            args=['/Users/hg/Documents/GitHub/mcp-server-sequential-thinking/dist/index.js'],
            env={
                'PATH': os.environ.get('PATH', ''),
                'NODE_ENV': 'production'
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
            """你是一个使用顺序思考方法的电网专家。
            你会将复杂问题分解成多个思考步骤，每个步骤都建立在前面的基础上。
            你可以修改之前的想法，开启新的思考分支，并在需要时增加更多的思考步骤。
            当撰写国内外研究现状时，
            使用时间线索，展现研究发展历程：  按照时间顺序描述研究进展，可以清晰地展现研究领域的演进过程。
            运用概括性动词，提炼研究核心内容：  使用精准的动词，概括描述不同研究的贡献，避免简单罗列。
            在提到研究成果时，添加参考文献（DOI：）
            只返回报告内容，不要返回任何解释。"""
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        self.conversation_history = messages.copy()  # 初始化时保存初始消息

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
            # 请求DeepSeek生成下一个思考步骤
            response = self.client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=8000
            )

            content = response.choices[0]
            self.conversation_history.append(content.message.model_dump())  # 保存AI的响应

            if content.finish_reason == "tool_calls":
                for tool_call in content.message.tool_calls:
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
                    tool_response = {
                        "role": "tool",
                        "content": result.content[0].text,
                        "tool_call_id": tool_call.id,
                    }
                    messages.append(tool_response)
                    # 保存工具响应
                    self.conversation_history.extend([content.message.model_dump(), tool_response])

                    # 如果不需要更多思考步骤，生成最终答案
                    if not result_data.get("nextThoughtNeeded", True):
                        final_response = self.client.chat.completions.create(
                            model="deepseek-v3-241226",
                            messages=messages,
                        )
                        answer = final_response.choices[0].message.content
                        self.conversation_history.append(final_response.choices[0].message.model_dump())  # 保存最终响应
                        
                        # 重置思考计数器，为下一次对话做准备
                        self.current_thought_number = 1
                        self.total_thoughts = 1
                        
                        return answer
            else:
                self.conversation_history.append(content.message.model_dump())  # 保存最终响应
                return content.message.content

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        query = """这是大纲：	## 3. 国内外研究水平综述
本章节将对国内外在电网配电AI领域的研究现状进行综述，分析技术发展历程及当前研究水平。通过对比国内外研究的差异与优势，为本项目的研究方向提供参考与借鉴。
### 3.1 技术发展历程
本节将回顾电网配电领域技术的发展历程，从早期的传统调度技术到现代的智能调度技术，分析重要技术突破与创新节点，探讨当前技术架构的特点及其对未来发展的影响。
### 3.2 国内外研究现状与对比
通过对国内外在电网配电AI领域的研究进展进行对比，分析各自的优势与不足。重点关注国内外在智能调度、故障诊断及负荷预测等方面的研究成果，为本项目的研究提供借鉴。
你会将复杂问题分解成多个思考步骤，每个步骤都建立在前面的基础上。
你可以修改之前的想法，开启新的思考分支，并在需要时增加更多的思考步骤。
根据大纲撰写一个国内外研究现状报告，每个段落1000字，至少3000字

"""
        response = await client.process_query(query)
        print("\n" + response)
        # 打印完整的对话历史
        # print("\n完整对话历史：")
        # print(json.dumps(client.conversation_history, indent=2, ensure_ascii=False))
    finally:
        await client.cleanup()

if __name__ == "__main__":
    # 启动web-search服务
    if os.environ.get('RUN_WEB_SEARCH', 'false').lower() == 'true':
        app.run(transport='stdio')
    else:
        asyncio.run(main())
