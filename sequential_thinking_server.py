#!/usr/bin/env python3

import json
import sys
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from colorama import init, Fore, Style

# 初始化 colorama 以支持跨平台的彩色输出
init()

@dataclass
class ThoughtData:
    """思考数据的数据类"""
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    is_revision: Optional[bool] = None
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: Optional[bool] = None

class SequentialThinkingServer:
    """顺序思考服务器类"""
    
    def __init__(self):
        self.thought_history: List[ThoughtData] = []
        self.branches: Dict[str, List[ThoughtData]] = {}

    def validate_thought_data(self, input_data: Any) -> ThoughtData:
        """验证输入的思考数据"""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        # 验证必填字段
        if not isinstance(input_data.get('thought'), str):
            raise ValueError("Invalid thought: must be a string")
        if not isinstance(input_data.get('thoughtNumber'), int):
            raise ValueError("Invalid thoughtNumber: must be a number")
        if not isinstance(input_data.get('totalThoughts'), int):
            raise ValueError("Invalid totalThoughts: must be a number")
        if not isinstance(input_data.get('nextThoughtNeeded'), bool):
            raise ValueError("Invalid nextThoughtNeeded: must be a boolean")

        # 创建并返回 ThoughtData 对象
        return ThoughtData(
            thought=input_data['thought'],
            thought_number=input_data['thoughtNumber'],
            total_thoughts=input_data['totalThoughts'],
            next_thought_needed=input_data['nextThoughtNeeded'],
            is_revision=input_data.get('isRevision'),
            revises_thought=input_data.get('revisesThought'),
            branch_from_thought=input_data.get('branchFromThought'),
            branch_id=input_data.get('branchId'),
            needs_more_thoughts=input_data.get('needsMoreThoughts')
        )

    def format_thought(self, thought_data: ThoughtData) -> str:
        """格式化思考步骤输出"""
        prefix = ''
        context = ''

        if thought_data.is_revision:
            prefix = f"{Fore.YELLOW}🔄 Revision{Style.RESET_ALL}"
            context = f" (revising thought {thought_data.revises_thought})"
        elif thought_data.branch_from_thought:
            prefix = f"{Fore.GREEN}🌿 Branch{Style.RESET_ALL}"
            context = f" (from thought {thought_data.branch_from_thought}, ID: {thought_data.branch_id})"
        else:
            prefix = f"{Fore.BLUE}💭 Thought{Style.RESET_ALL}"
            context = ''

        header = f"{prefix} {thought_data.thought_number}/{thought_data.total_thoughts}{context}"
        max_length = max(len(header), len(thought_data.thought))
        border = "─" * (max_length + 4)

        return f"""
┌{border}┐
│ {header.ljust(max_length + 2)} │
├{border}┤
│ {thought_data.thought.ljust(max_length + 2)} │
└{border}┘"""

    async def process_thought(self, input_data: Any) -> Dict[str, Any]:
        """处理思考步骤输入并返回结果"""
        try:
            validated_input = self.validate_thought_data(input_data)

            # 更新总思考数如果需要
            if validated_input.thought_number > validated_input.total_thoughts:
                validated_input.total_thoughts = validated_input.thought_number

            # 添加到历史记录
            self.thought_history.append(validated_input)

            # 处理分支
            if validated_input.branch_from_thought and validated_input.branch_id:
                if validated_input.branch_id not in self.branches:
                    self.branches[validated_input.branch_id] = []
                self.branches[validated_input.branch_id].append(validated_input)

            # 输出格式化的思考步骤
            print(self.format_thought(validated_input), file=sys.stderr)

            # 返回处理结果
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "thoughtNumber": validated_input.thought_number,
                        "totalThoughts": validated_input.total_thoughts,
                        "nextThoughtNeeded": validated_input.next_thought_needed,
                        "branches": list(self.branches.keys()),
                        "thoughtHistoryLength": len(self.thought_history)
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": str(e),
                        "status": "failed"
                    }, indent=2)
                }],
                "isError": True
            }

class MCPServer:
    """MCP 服务器类"""
    
    def __init__(self):
        self.thinking_server = SequentialThinkingServer()
        self.sequential_thinking_tool = {
            "name": "sequentialthinking",
            "description": """A detailed tool for dynamic and reflective problem-solving through thoughts.
This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
Each thought can build on, question, or revise previous insights as understanding deepens.

When to use this tool:
- Breaking down complex problems into steps
- Planning and design with room for revision
- Analysis that might need course correction
- Problems where the full scope might not be clear initially

Key features:
- Adjustable total thoughts
- Revision capability
- Branching support
- Dynamic thought process""",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Your current thinking step"},
                    "nextThoughtNeeded": {"type": "boolean", "description": "Whether another thought step is needed"},
                    "thoughtNumber": {"type": "integer", "description": "Current thought number", "minimum": 1},
                    "totalThoughts": {"type": "integer", "description": "Estimated total thoughts needed", "minimum": 1},
                    "isRevision": {"type": "boolean", "description": "Whether this revises previous thinking"},
                    "revisesThought": {"type": "integer", "description": "Which thought is being reconsidered", "minimum": 1},
                    "branchFromThought": {"type": "integer", "description": "Branching point thought number", "minimum": 1},
                    "branchId": {"type": "string", "description": "Branch identifier"},
                    "needsMoreThoughts": {"type": "boolean", "description": "If more thoughts are needed"}
                },
                "required": ["thought", "nextThoughtNeeded", "thoughtNumber", "totalThoughts"]
            }
        }

    async def handle_request(self, request: str) -> str:
        """处理 MCP 请求"""
        try:
            request_data = json.loads(request)
            if request_data.get("method") == "list_tools":
                return json.dumps({"tools": [self.sequential_thinking_tool]})
            elif request_data.get("method") == "call_tool":
                params = request_data.get("params", {})
                if params.get("name") == "sequentialthinking":
                    result = await self.thinking_server.process_thought(params.get("arguments", {}))
                    return json.dumps(result)
                else:
                    return json.dumps({
                        "content": [{
                            "type": "text",
                            "text": f"Unknown tool: {params.get('name')}"
                        }],
                        "isError": True
                    })
            else:
                return json.dumps({
                    "content": [{
                        "type": "text",
                        "text": f"Unknown method: {request_data.get('method')}"
                    }],
                    "isError": True
                })
        except Exception as e:
            return json.dumps({
                "content": [{
                    "type": "text",
                    "text": f"Error processing request: {str(e)}"
                }],
                "isError": True
            })

async def main():
    """主函数"""
    print("Sequential Thinking MCP Server running on stdio", file=sys.stderr)
    server = MCPServer()
    
    try:
        while True:
            request = await asyncio.get_event_loop().run_in_executor(None, input)
            if not request:
                continue
            response = await server.handle_request(request)
            print(response, flush=True)
    except KeyboardInterrupt:
        print("\nServer shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"Fatal error running server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())