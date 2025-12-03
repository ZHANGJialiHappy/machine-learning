import json
import csv
import boto3
from botocore.config import Config

# AWS Bedrock 配置
BEDROCK_REGION = "eu-north-1"  # 根据您的区域调整
# Claude 4.5 在 eu-north-1 区域的模型ID
MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"  # Claude 4.5

# Prompt模板
PROMPT_TEMPLATE = """You are an expert customer service analyst for Blackbird Coffee's support operations.  

Always respond with a SINGLE valid JSON object only, with no additional text, comments, or explanation outside the JSON object.  

Do NOT use Markdown formatting.

CONTEXT - Understanding the Business Process:

Blackbird Coffee provides customer support for coffee machine installations, maintenance, and general inquiries. 

After each customer service call, we need to:

1) Understand what happened in the call (Summary)

2) Identify ALL topics discussed (a call may cover multiple issues)

3) Determine what actions the support team should take next for each issue

The call you are analyzing is a customer service interaction where a Blackbird Coffee support representative 

is responding to a customer inquiry or issue.

CALL TRANSCRIPT (complete conversation without timestamps):

%s

ANALYSIS INSTRUCTIONS:

The transcript is a mixed conversation where both speakers' words are combined in chronological order.

Analyze the conversation and produce the following fields in the JSON response:

1. CONVERSATION SUMMARY (string)

   - Provide a concise summary (2-4 sentences) covering:

     - What were the customer's issues or inquiries? (list all if multiple)

     - What did the support representative do or explain?

     - What was the outcome or current status?

     - Any important details (e.g., serial numbers, dates, promises made, follow-up commitments)

   - The summary MUST be written in English, even if the conversation is in Danish.

   - Be factual and objective. Do not add information that is not in the transcript.

2. TOPIC CLASSIFICATION (array of enums)

   - List ALL topics discussed in this conversation. A single call may cover multiple topics.

   - Use ONLY the following enum values:

   - COFFEE_MACHINE_NOT_WORKING: Machine is broken, malfunctioning, or not operating correctly
   - RESTOCK_SUPPLIES: Customer is missing coffee powder, chocolate powder, milk powder, cleaning supplies or other supplies
   - QUESTION_ABOUT_INVOICE: Customer has questions about billing or invoice details
   - QUESTION_ABOUT_CONTRACT: Customer has questions about the contract or agreement
   - AUTOMATION_FAILURE: Issues where automated company systems (e.g., replenishment subscriptions, delivery scheduling)
   - COMPLAINT: Customer is making a formal complaint about service, product, or experience
   - GENERAL_INQUIRY: General questions or information requests
   - OTHER: Does not fit any of the above categories

   - Return an array of enum values, e.g.: ["COFFEE_MACHINE_NOT_WORKING", "MISSING_COFFEE_OR_DRINKS"]

   - If only one topic is discussed, still return an array with a single element.

3. NEXT ACTIONS (array of objects)

   - List ALL actions that need to be taken based on this conversation.

   - A single call may require multiple follow-up actions.

   - If no further action is required, return a single action with action = "NO_ACTION_REQUIRED", urgency = "NONE".

   For each action object, include:

   a) action (ACTION CATEGORY, string - enum)

      Use ONLY one of:

      - NO_ACTION_REQUIRED: Issue resolved, no follow-up needed
      - SCHEDULE_TECHNICIAN_VISIT: Need to send a technician to customer location
      - INTERNAL_SYSTEM_CORRECTION: Fixing an error within the company internal systems (e.g., billing, automation).
      - SEND_MATERIALS_SUPPLIES: Need to deliver coffee, drinks, or care products
      - DOCUMENT_AND_FOLLOW_UP: Need to call customer back (e.g., waiting for information, promised callback)
      - ESCALATE_TO_SPECIALIST_TEAM: Issue needs manager or specialist team review or approval
      - UPDATE_INVOICE: Need to adjust billing or send corrected invoice
      - REMOTE_SUPPORT: Need to provide remote machine diagnostics or configuration
      - OTHER: Other action needed (specify in details)

   b) details (string)

      - 1-2 sentences describing what exactly should be done.

      - Clearly state which topic/issue this action relates to.

      - Include any deadlines or urgency mentioned.

      - Include key information where available (e.g., address, part numbers, machine type, customer preferences).

   c) urgency (URGENCY LEVEL, string - enum)

      Use ONLY one of:

      - CRITICAL: Customer is completely blocked, immediate action required (within hours)

      - HIGH: Significant impact, action needed within 1-2 business days

      - MEDIUM: Moderate impact, action needed within 1 week

      - LOW: Minor issue, can be handled in the normal course of business

      - NONE: No action required or no urgency

      If urgency is unclear, default to "MEDIUM".

4. CUSTOMER, AGENT, COMPANY FIELDS

   - customer: The name of the customer, if mentioned. If not mentioned, use null.

   - agent: The name of the Blackbird Coffee support employee, if mentioned. If not mentioned, use null.

   - company: The customer's company name, if mentioned. If not mentioned, use null.

   - Do NOT invent or guess names or company information. Only use names/companies explicitly stated or clearly implied in the transcript.

RESPONSE FORMAT (JSON ONLY):

Return a single JSON object with EXACTLY the following structure and keys:

{

  "summary": "<2-4 sentence summary in English>",

  "topics": ["<ENUM_VALUE1>", "<ENUM_VALUE2>", ...],

  "next_actions": [

    {

      "action": "<ACTION_CATEGORY>",

      "details": "<Specific details about what needs to be done>",

      "urgency": "<URGENCY_LEVEL>"

    }

    // Additional actions as needed

  ],

  "customer": "<customer name or null>",

  "agent": "<agent name or null>",

  "company": "<company name or null>"

}

CRITICAL REMINDERS:

- Return ONLY a single valid JSON object. No extra text before or after the JSON.

- Do NOT use Markdown formatting.

- The transcript may be in Danish - analyze it carefully, but always write the summary in English.

- "topics" MUST be an array, even if only one topic.

- "next_actions" MUST be an array, even if only one action.

- Be specific in "details": include what, when, who, and which issue it relates to.

- If multiple topics are discussed, list ALL of them in "topics".

- If multiple actions are needed, list ALL of them in "next_actions".

- Each action's "details" should clearly reference the relevant topic/issue.

- If no customer name, agent name, or company name is mentioned, set the corresponding field to null.

- If urgency is unclear, default to "MEDIUM"."""


def invoke_bedrock_model(transcript):

    try:
        # 创建Bedrock运行时客户端
        config = Config(
            region_name=BEDROCK_REGION,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        bedrock_runtime = boto3.client('bedrock-runtime', config=config)
        
        # 准备提示词
        prompt = PROMPT_TEMPLATE % transcript
        
        # 使用InvokeModel API（与用户成功代码相同）
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",  # Claude必需字段
            "max_tokens": 2000,
            "temperature": 0.0,
            "system": "You are an expert customer service analyst. Always return valid JSON only.",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }
        
        # 调用Bedrock
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_body)
        )
        
        # 解析响应（与用户成功代码相同）
        response_body = json.loads(response['body'].read())
        
        # 提取Claude的回答
        content_text = response_body['content'][0]['text']
        
        # 清理可能的Markdown格式
        content_text = content_text.strip()
        if content_text.startswith('```json'):
            content_text = content_text[7:]
        if content_text.startswith('```'):
            content_text = content_text[3:]
        if content_text.endswith('```'):
            content_text = content_text[:-3]
        content_text = content_text.strip()
        
        # 解析JSON
        result = json.loads(content_text)
        
        # 记录token使用
        if 'usage' in response_body:
            print(f"  ℹ️  Tokens - Input: {response_body['usage'].get('input_tokens', 0)}, "
                  f"Output: {response_body['usage'].get('output_tokens', 0)}")
        
        return result
            
    except Exception as e:
        print(f"Error invoking Bedrock model: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_transcriptions(input_file, output_file):

    print(f"Reading transcriptions from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        transcriptions = json.load(f)
    
    
    print(f"Found {len(transcriptions)} transcriptions to process.")
    
    # 准备CSV输出
    results = []
    
    # 处理每条记录
    for idx, record in enumerate(transcriptions, 1):
        print(f"\nProcessing record {idx}/{len(transcriptions)}...")
        
        transcript = record.get('complete_text', '')
        if not transcript:
            print(f"Warning: Record {idx} has no complete_text, skipping.")
            continue
        
        # 调用模型分析
        analysis = invoke_bedrock_model(transcript)
        
        if analysis:
            # 准备CSV行
            row = {
                'Call_ID': f'CALL_{idx:04d}',
                'complete_text': transcript,
                'summary': analysis.get('summary', ''),
                'topics': json.dumps(analysis.get('topics', [])),  # 转换为JSON字符串
                'customer': analysis.get('customer', ''),
                'agent': analysis.get('agent', ''),
                'company': analysis.get('company', ''),
                'next_actions': json.dumps(analysis.get('next_actions', []))  # 转换为JSON字符串
            }
            results.append(row)
            print(f"✓ Successfully processed record {idx}")
        else:
            print(f"✗ Failed to process record {idx}")
            # 添加失败记录（保留原文本）
            row = {
                'Call_ID': f'CALL_{idx:04d}',
                'complete_text': transcript,
                'summary': 'ERROR: Failed to process',
                'topics': '[]',
                'customer': '',
                'agent': '',
                'company': '',
                'next_actions': '[]'
            }
            results.append(row)
    
    # 写入CSV文件
    print(f"\nWriting results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['Call_ID', 'complete_text', 'summary', 'topics', 
                     'customer', 'agent', 'company', 'next_actions']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Processing complete! Results saved to {output_file}")
    print(f"Total records processed: {len(results)}")


if __name__ == "__main__":
    # 文件路径
    input_file = "data/blackbirddb_public_call_transcription.json"
    output_file = "customer_service_analysis_results.csv"
    
    # 处理记录
    process_transcriptions(input_file, output_file)

