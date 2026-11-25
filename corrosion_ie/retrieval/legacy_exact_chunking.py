"""
科研论文 Markdown 文档切分模块
针对 RAG 问答系统优化，实现：
1. 表格提取：精准定位 Markdown/HTML 表格，封装为独立 Document
2. 章节切分：按 Header 切分大块
3. 递归细分：按 600 字符递归切分，保护 LaTeX 公式完整性
4. 元数据关联：与 metadata_index.json 中的文档元数据关联
"""

import re
import json
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)


class CustomMarkdownSplitter:
    """
    针对科研论文优化的 Markdown 切分器
    
    特点：
    1. 表格作为独立块保护，不会被切分
    2. 按章节(#, ##, ###)层次化切分
    3. 保护 LaTeX 公式 ($...$, $$...$$) 完整性
    4. 支持元数据关联
    """
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Markdown 表格正则：匹配 | col | col | 格式
        # 匹配表头行 + 分隔行 + 数据行
        self.md_table_pattern = re.compile(
            r'((?:^|\n)\|[^\n]+\|\s*\n'      # 表头行
            r'\|[\s:\-|]+\|\s*\n'             # 分隔行 (|---|---|)
            r'(?:\|[^\n]+\|\s*\n?)*)',        # 数据行（可多行）
            re.MULTILINE
        )
        
        # HTML 表格正则 - 同时捕获表格前的标题行（如 **Table 3 ...** 或 Table 3: ...）
        self.html_table_pattern = re.compile(
            r'(\*{0,2}Table\s*\d+[^\n]*\*{0,2}\n)?(<table>.*?</table>)',
            re.DOTALL | re.IGNORECASE
        )
        
        # 带标签的表格块（如 === Table 1 === 格式）
        self.labeled_table_pattern = re.compile(
            r'(===\s*Table\s*\d+\s*===.*?(?=\n===|\n##|\n#|\Z))',
            re.DOTALL
        )
        
        # 章节分隔器配置
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ]
        
        # 递归切分的分隔符（按优先级排序 - 语义优先）
        # 优先保持段落完整，其次是句子完整，最后才是字符级切分
        self.separators = [
            "\n\n\n",             # 1. 多个空行（段落间隔）
            "\n\n",               # 2. 双换行（段落）
            r"(?<=。)\s*",         # 3. 中文句号后
            r"(?<![A-Z])(?<=\.)\s+(?![0-9a-z])", # 4. 英文句号后（排除缩写如 A. terreus, Fig. 1, et al.）
            r"(?<=！)\s*",         # 5. 中文感叹号后
            r"(?<=？)\s*",         # 6. 中文问号后
            r"(?<=\?)\s+",        # 7. 英文问号后
            "\n",                 # 8. 单换行
            r"(?<=；)\s*",         # 9. 中文分号后
            r"(?<=;)\s+",         # 10. 英文分号后
            r"(?<=，)\s*",         # 11. 中文逗号后
            r"(?<=,)\s+",         # 12. 英文逗号后
            " ",                   # 13. 空格
            "",                    # 14. 字符级（最后手段）
        ]
        
        # 是否使用正则分隔符
        self.is_separator_regex = True
        
        # 短块合并的最小长度阈值（低于此值的块会尝试与下一块合并）
        # 提高到200以减少碎片化短块，提升embedding质量
        self.min_chunk_length = 200
        
        # 最终过滤的最小块长度（低于此值的块将被丢弃）
        self.min_final_chunk_length = 80
        
        # 是否添加标题路径前缀（Parent Document Retrieval 策略）
        self.add_header_path = True
    
    def _remove_overlap_duplicates(self, content: str, prev_chunk_tail: str = "") -> str:
        """
        消除由 overlap 导致的重复句子
        
        仅对当前 chunk 开头与上一个 chunk 末尾进行重叠比对，
        而不是对整个 chunk 内部去重（避免误删文档中本身重复的内容）
        
        Args:
            content: 当前 chunk 内容
            prev_chunk_tail: 上一个 chunk 的末尾文本（用于比对）
            
        Returns:
            去除开头重叠部分后的内容
        """
        if not content or not prev_chunk_tail:
            return content
        
        # 提取当前 chunk 开头的文本（最多 300 字符，用于比对）
        compare_length = min(300, self.chunk_overlap + 100)
        content_head = content[:compare_length]
        
        # 提取上一个 chunk 末尾的文本
        tail_length = min(300, len(prev_chunk_tail))
        prev_tail = prev_chunk_tail[-tail_length:]
        
        # 寻找重叠部分：从最长可能的重叠开始尝试
        max_overlap = min(len(content_head), len(prev_tail))
        overlap_start = 0
        
        for overlap_len in range(max_overlap, 20, -1):  # 至少20字符才算有效重叠
            # 检查 content 开头是否与 prev_tail 末尾重叠
            if content_head[:overlap_len] == prev_tail[-overlap_len:]:
                overlap_start = overlap_len
                break
        
        if overlap_start > 0:
            # 找到重叠，移除 content 开头的重叠部分
            # 但要保留完整的行（避免切断句子中间）
            trimmed = content[overlap_start:]
            # 如果切断了行，找到下一个换行符
            if trimmed and not content[overlap_start - 1:overlap_start] in ('\n', ''):
                newline_pos = trimmed.find('\n')
                if newline_pos != -1 and newline_pos < 50:  # 在50字符内找到换行
                    trimmed = trimmed[newline_pos + 1:]
            return trimmed.lstrip('\n')  # 移除开头多余的换行
        
        return content
    
    def _build_header_path(self, metadata: Dict) -> str:
        """
        构建标题路径前缀，用于增强语义明确性
        
        格式: [Title > Section > Subsection]
        
        策略：
        - header_1 或 title (论文标题): 截断到 80 字符
        - header_2-4 (章节标题): 不截断（通常较短）
        """
        path_parts = []
        
        # 优先使用 header_1，如果没有则使用 title（用于表格等没有经过章节解析的块）
        header_1 = metadata.get('header_1') or metadata.get('title', '')
        if header_1:
            header_1 = header_1.strip()
            if len(header_1) > 80:
                header_1 = header_1[:77] + '...'
            path_parts.append(header_1)
        
        # 添加其他层级标题
        for key in ['header_2', 'header_3', 'header_4']:
            if key in metadata and metadata[key]:
                header = metadata[key].strip()
                path_parts.append(header)
        
        if path_parts:
            return '[' + ' > '.join(path_parts) + ']'
        return ''
    
    def _is_header_only(self, text: str) -> bool:
        """检查文本是否只包含标题（没有实质内容）"""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if not lines:
            return True
        # 所有非空行都是标题
        return all(line.startswith('#') for line in lines)
    
    def _is_meaningless_chunk(self, text: str) -> bool:
        """
        检查是否是无意义的块（应该被过滤掉）
        
        只过滤纯表格引用占位符块（表格内容已在独立表格块中保留）
        """
        text = text.strip()
        
        # 过滤纯表格引用（单个或多个，可能带标题路径前缀）
        if re.match(r'^(\[.*?\]\s*)?(##?\s*表格数据\s*)?(\[参见表格\s*\d+\]\s*)+$', text, re.MULTILINE | re.DOTALL):
            return True
        
        return False

    def _extract_tables(self, text: str) -> Tuple[str, List[Dict]]:
        """
        提取文档中的所有表格，返回处理后的文本和表格列表
        
        Args:
            text: 原始文档文本
            
        Returns:
            (处理后的文本, 表格信息列表)
        """
        tables = []
        
        # 步骤1：提取带标签的表格块（如 "=== Table 1 ===" 格式）
        def replace_labeled_table(match):
            table_content = match.group(1).strip()
            
            # 从表格内容中提取真实编号（如 "Table 1", "Table 2" 等）
            real_label = None
            # 匹配 "=== Table X ===" 或 "label: Table X"
            label_match = re.search(r'===\s*(Table\s*\d+)\s*===', table_content)
            if label_match:
                real_label = label_match.group(1)  # 如 "Table 1"
            else:
                # 尝试从 label 行提取
                label_line_match = re.search(r'label:\s*(Table\s*\d+)', table_content, re.IGNORECASE)
                if label_line_match:
                    real_label = label_line_match.group(1)
            
            tables.append({
                'content': table_content,
                'type': 'labeled_table',
                'index': len(tables),
                'real_label': real_label  # 保存真实编号
            })
            return f"\n__TABLE_PLACEHOLDER_{len(tables)-1}__\n"
        
        text = self.labeled_table_pattern.sub(replace_labeled_table, text)
        
        # 步骤2：提取 HTML 表格
        def replace_html_table(match):
            title_line = match.group(1)  # 可能是 None 或 "**Table 3 ...**\n"
            table_content = match.group(2).strip()
            
            # 尝试从标题行提取真实编号
            real_label = None
            if title_line:
                label_match = re.search(r'Table\s*(\d+)', title_line, re.IGNORECASE)
                if label_match:
                    real_label = f"Table {label_match.group(1)}"
                # 将标题行也加入表格内容
                table_content = title_line.strip() + '\n' + table_content
            
            tables.append({
                'content': table_content,
                'type': 'html_table',
                'index': len(tables),
                'real_label': real_label
            })
            return f"\n__TABLE_PLACEHOLDER_{len(tables)-1}__\n"
        
        text = self.html_table_pattern.sub(replace_html_table, text)
        
        # 步骤3：提取 Markdown 管道符表格
        def replace_md_table(match):
            table_content = match.group(1).strip()
            if table_content:  # 确保不是空匹配
                tables.append({
                    'content': table_content,
                    'type': 'markdown_table',
                    'index': len(tables)
                })
                return f"\n__TABLE_PLACEHOLDER_{len(tables)-1}__\n"
            return match.group(0)
        
        text = self.md_table_pattern.sub(replace_md_table, text)
        
        return text, tables

    def _protect_formulas(self, text: str) -> Tuple[str, List[str]]:
        """
        保护 LaTeX 公式，防止被切分器在公式内部切断
        
        将 $...$ 和 $$...$$ 替换为占位符，切分后再还原
        
        Args:
            text: 原始文本
            
        Returns:
            (处理后的文本, 公式列表)
        """
        formulas = []
        
        # 先处理 $$...$$ (块级公式)，避免与 $...$ 冲突
        def replace_block_formula(match):
            formula = match.group(0)
            formulas.append(formula)
            # 使用不含特殊字符的占位符
            return f"__FORMULA_{len(formulas)-1}__"
        
        # 匹配 $$...$$，允许跨行
        text = re.sub(r'\$\$.*?\$\$', replace_block_formula, text, flags=re.DOTALL)
        
        # 再处理 $...$ (行内公式)
        def replace_inline_formula(match):
            formula = match.group(0)
            formulas.append(formula)
            return f"__FORMULA_{len(formulas)-1}__"
        
        # 匹配 $...$，不跨行，内部不能有 $
        # 使用非贪婪匹配，确保匹配最短的公式
        text = re.sub(r'\$[^$\n]+?\$', replace_inline_formula, text)
        
        return text, formulas
    
    def _restore_formulas(self, text: str, formulas: List[str]) -> str:
        """
        还原被保护的公式
        
        Args:
            text: 含占位符的文本
            formulas: 公式列表
            
        Returns:
            还原后的文本
        """
        for i, formula in enumerate(formulas):
            text = text.replace(f"__FORMULA_{i}__", formula)
        return text

    def _check_formula_integrity(self, text: str) -> bool:
        """
        检查文本中的 LaTeX 公式是否完整
        
        Args:
            text: 待检查的文本
            
        Returns:
            True 如果公式完整，False 如果公式被截断
        """
        # 统计 $$ 和 $ 的数量
        double_dollar_count = text.count('$$')
        
        # 移除 $$ 后统计单独的 $
        text_without_double = text.replace('$$', '')
        single_dollar_count = text_without_double.count('$')
        
        # $$ 必须成对，$ 也必须成对
        return double_dollar_count % 2 == 0 and single_dollar_count % 2 == 0

    def _merge_broken_formulas(self, chunks: List[Document]) -> List[Document]:
        """
        合并被截断的公式块
        
        修复策略：
        1. 只合并相邻的少量块（最多合并 3 个）
        2. 合并后长度不超过 chunk_size * 2
        3. 防止因为持续的公式不完整导致无限合并
        
        Args:
            chunks: 原始切分的文档块列表
            
        Returns:
            合并后的文档块列表
        """
        if not chunks:
            return chunks
        
        merged = []
        i = 0
        
        # 最大合并限制
        max_merge_count = 3  # 最多合并 3 个连续块
        max_merged_length = self.chunk_size * 2  # 合并后最大长度
        
        while i < len(chunks):
            current = chunks[i]
            content = current.page_content
            merge_count = 0
            
            # 如果当前块公式不完整，尝试与后续块合并
            # 但限制合并次数和长度
            while (not self._check_formula_integrity(content) 
                   and i + 1 < len(chunks) 
                   and merge_count < max_merge_count
                   and len(content) < max_merged_length):
                i += 1
                merge_count += 1
                next_chunk = chunks[i]
                content = content + "\n" + next_chunk.page_content
                # 合并元数据（保留当前块的元数据）
            
            # 创建新的 Document
            merged.append(Document(
                page_content=content,
                metadata=current.metadata.copy()
            ))
            i += 1
        
        return merged

    def _remove_duplicate_headers(self, content1: str, content2: str) -> str:
        """
        移除合并时重复的标题
        
        当合并两个块时，如果 content1 的末尾标题与 content2 的开头标题相同，
        则去除重复。
        
        Args:
            content1: 第一个块的内容
            content2: 第二个块的内容
            
        Returns:
            合并后去重的内容
        """
        lines1 = content1.strip().split('\n')
        lines2 = content2.strip().split('\n')
        
        if not lines1 or not lines2:
            return content1.strip() + "\n\n" + content2.strip()
        
        # 获取 content1 末尾的标题行
        header_pattern = re.compile(r'^#{1,6}\s+')
        trailing_headers = []
        for line in reversed(lines1):
            line = line.strip()
            if header_pattern.match(line):
                trailing_headers.insert(0, line)
            elif line:  # 遇到非空非标题行，停止
                break
        
        # 获取 content2 开头的标题行
        leading_headers = []
        for line in lines2:
            line = line.strip()
            if header_pattern.match(line):
                leading_headers.append(line)
            elif line:  # 遇到非空非标题行，停止
                break
        
        # 如果有重复的标题，从 content2 中移除
        if trailing_headers and leading_headers:
            # 找到重叠的标题
            overlap_count = 0
            for i, h in enumerate(leading_headers):
                if i < len(trailing_headers) and h == trailing_headers[i]:
                    overlap_count += 1
                else:
                    break
            
            # 也检查完全相同的情况
            if overlap_count == 0:
                for th in trailing_headers:
                    if th in leading_headers:
                        overlap_count = max(overlap_count, 1)
                        break
            
            if overlap_count > 0:
                # 从 content2 中移除重复的标题行
                new_lines2 = []
                skipped = 0
                for line in lines2:
                    line_stripped = line.strip()
                    if skipped < len(leading_headers) and line_stripped in trailing_headers:
                        skipped += 1
                        continue
                    new_lines2.append(line)
                content2 = '\n'.join(new_lines2)
        
        return content1.strip() + "\n\n" + content2.strip()

    def _merge_short_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        合并过短的文本块
        
        核心规则：
        1. 以#开头的块只能向下合并，永远不向上合并
        2. 普通块如果太短，优先向上合并（如果下一块是标题块）
        3. 否则向下合并
        
        这样确保标题始终在块的开头
        
        Args:
            chunks: 原始切分的文档块列表
            
        Returns:
            处理后的文档块列表
        """
        if not chunks:
            return chunks
        
        merged = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            content = current.page_content.strip()
            meta = current.metadata.copy()
            
            # 检查当前块是否以标题开头
            first_line = content.split('\n')[0].strip() if content else ""
            starts_with_header = first_line.startswith('#')
            
            # 检查是否只有标题或过短
            is_header_only = self._is_header_only(content)
            is_short = len(content) < self.min_chunk_length
            
            # 如果需要合并（太短或纯标题）
            if is_header_only or is_short:
                # 检查下一块是否以标题开头
                next_starts_with_header = False
                if i + 1 < len(chunks):
                    next_first_line = chunks[i + 1].page_content.strip().split('\n')[0].strip()
                    next_starts_with_header = next_first_line.startswith('#')
                
                # 如果当前块不是标题块，且下一块是标题块，则向上合并
                if not starts_with_header and next_starts_with_header and merged:
                    last = merged[-1]
                    if len(last.page_content) + len(content) <= self.chunk_size * 1.5:
                        merged[-1] = Document(
                            page_content=last.page_content + "\n\n" + content,
                            metadata=last.metadata
                        )
                        i += 1
                        continue
                
                # 否则向下合并（如果可以）
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    combined = self._remove_duplicate_headers(content, next_chunk.page_content)
                    
                    # 合并元数据
                    for key in ['header_1', 'header_2', 'header_3', 'header_4']:
                        if key in next_chunk.metadata and next_chunk.metadata[key]:
                            if key not in meta or not meta[key]:
                                meta[key] = next_chunk.metadata[key]
                    
                    # 继续向下合并直到足够长（但不合并标题块）
                    while (self._is_header_only(combined) or len(combined) < self.min_chunk_length) and i + 2 < len(chunks):
                        # 检查再下一块是否是标题块
                        next_next_first = chunks[i + 2].page_content.strip().split('\n')[0].strip()
                        if next_next_first.startswith('#'):
                            break  # 不合并标题块
                        
                        i += 1
                        next_chunk = chunks[i + 1]
                        combined = self._remove_duplicate_headers(combined, next_chunk.page_content)
                        for key in ['header_1', 'header_2', 'header_3', 'header_4']:
                            if key in next_chunk.metadata and next_chunk.metadata[key]:
                                if key not in meta or not meta[key]:
                                    meta[key] = next_chunk.metadata[key]
                        if len(combined) > self.chunk_size * 1.5:
                            break
                    
                    merged.append(Document(page_content=combined, metadata=meta))
                    i += 2
                    continue
            
            # 不需要合并
            merged.append(current)
            i += 1
        
        return merged

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        切分文档列表
        
        Args:
            documents: 待切分的 Document 列表
            
        Returns:
            切分后的 Document 列表
        """
        final_docs = []
        
        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata.copy()
            
            # --- 步骤 A: 提取所有表格 ---
            text_without_tables, tables = self._extract_tables(text)
            
            # --- 步骤 B: 为表格创建独立 Document（暂存，稍后按原文位置插入） ---
            table_docs = {}  # placeholder_index -> Document
            for tbl in tables:
                table_meta = metadata.copy()
                table_meta['content_type'] = 'table'
                table_meta['table_format'] = tbl['type']
                table_meta['table_index'] = tbl['index']
                # 添加真实表格编号（如 "Table 1"）
                if tbl.get('real_label'):
                    table_meta['table_label'] = tbl['real_label']
                table_meta['chunk_origin'] = 'table_extraction'
                
                # 标题路径存入 metadata（方案A：不污染 content，保持向量纯净）
                table_content = tbl['content']
                if self.add_header_path:
                    header_path = self._build_header_path(table_meta)
                    if header_path:
                        table_meta['header_path'] = header_path  # 存入 metadata
                
                table_docs[tbl['index']] = Document(
                    page_content=table_content,
                    metadata=table_meta
                )
            
            # --- 步骤 C: 保护公式，防止被切分 ---
            text_protected, formulas = self._protect_formulas(text_without_tables)
            
            # --- 步骤 D: 按章节切分 ---
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split_on,
                strip_headers=False  # 保留标题
            )
            
            try:
                header_splits = markdown_splitter.split_text(text_protected)
            except Exception as e:
                # 如果章节切分失败，将整个文档作为一个块
                header_splits = [Document(
                    page_content=text_protected,
                    metadata={}
                )]
            
            # --- 步骤 E: 递归字符切分 ---
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                is_separator_regex=self.is_separator_regex
            )
            
            recursive_splits = text_splitter.split_documents(header_splits)
            
            # --- 步骤 F: 还原公式 ---
            for split_doc in recursive_splits:
                split_doc.page_content = self._restore_formulas(split_doc.page_content, formulas)
            
            # --- 步骤 G: 合并被截断的公式（作为后备保护）---
            recursive_splits = self._merge_broken_formulas(recursive_splits)
            
            # --- 步骤 G2: 合并过短的碎片块 ---
            recursive_splits = self._merge_short_chunks(recursive_splits)
            
            # --- 步骤 H: 处理元数据、清理占位符，并按原文位置插入表格 ---
            inserted_tables = set()  # 已插入的表格占位符索引
            prev_chunk_tail = ""  # 用于去重比对的上一个 chunk 末尾
            for split_doc in recursive_splits:
                combined_meta = {**metadata, **split_doc.metadata}
                combined_meta['content_type'] = 'text'
                combined_meta['chunk_origin'] = 'text_splitting'
                
                content = split_doc.page_content
                
                # 检测占位符并在此位置插入对应的表格 Document
                placeholder_indices = [
                    int(m.group(1))
                    for m in re.finditer(r'__TABLE_PLACEHOLDER_(\d+)__', content)
                ]
                for tbl_idx in placeholder_indices:
                    if tbl_idx not in inserted_tables and tbl_idx in table_docs:
                        final_docs.append(table_docs[tbl_idx])
                        inserted_tables.add(tbl_idx)
                
                # 替换表格占位符为提示信息
                if "__TABLE_PLACEHOLDER_" in content:
                    content = re.sub(
                        r'__TABLE_PLACEHOLDER_(\d+)__',
                        r'[参见表格 \1]',
                        content
                    )
                
                # 消除 overlap 造成的重复句子（仅比对与前一个 chunk 的重叠区域）
                content = self._remove_overlap_duplicates(content, prev_chunk_tail)
                
                # 标题路径存入 metadata（方案A：不污染 content，保持向量纯净）
                # 检索返回时再拼接，节省 embedding token，提升向量质量
                if self.add_header_path:
                    header_path = self._build_header_path(combined_meta)
                    if header_path:
                        combined_meta['header_path'] = header_path  # 存入 metadata
                
                final_docs.append(Document(
                    page_content=content.strip(),
                    metadata=combined_meta
                ))
                
                # 保存当前 chunk 末尾用于下一个 chunk 的去重比对
                prev_chunk_tail = split_doc.page_content
            
            # 将未被任何文本块引用的表格追加到末尾（兜底）
            for tbl_idx in sorted(table_docs.keys()):
                if tbl_idx not in inserted_tables:
                    final_docs.append(table_docs[tbl_idx])
        
        # --- 步骤 I: 过滤无意义的块 ---
        # 1. 过滤掉纯表格引用、过短内容等无意义块
        # 2. 表格块不过滤（它们有独立价值）
        filtered_docs = []
        for doc in final_docs:
            content_type = doc.metadata.get('content_type', 'text')
            
            # 表格块保留
            if content_type == 'table':
                filtered_docs.append(doc)
                continue
            
            # 文本块检查是否有意义
            if not self._is_meaningless_chunk(doc.page_content):
                filtered_docs.append(doc)
        
        # --- 步骤 J: 添加检索增强元数据 ---
        # 按 doc_id (source) 分组编号，避免窗口扩展时跨文档串台
        from collections import defaultdict
        
        # 第一遍：按 source 分组
        docs_by_source = defaultdict(list)
        for doc in filtered_docs:
            source = doc.metadata.get('source', 'unknown')
            docs_by_source[source].append(doc)
        
        # 第二遍：为每个文档内的 chunks 编号
        for source, doc_chunks in docs_by_source.items():
            total_in_doc = len(doc_chunks)
            for idx, doc in enumerate(doc_chunks):
                # 文档内编号（用于窗口扩展）
                doc.metadata['chunk_index_in_doc'] = idx
                doc.metadata['total_chunks_in_doc'] = total_in_doc
                
                # 保留全局编号（向后兼容）
                doc.metadata['chunk_index'] = idx  # 现在等于 chunk_index_in_doc
                doc.metadata['total_chunks'] = total_in_doc  # 现在等于 total_chunks_in_doc
                
                # 生成唯一 chunk_id: {source}#chunk_{index_in_doc}
                doc.metadata['chunk_id'] = f"{source}#chunk_{idx}"
        
        return filtered_docs

    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        切分单个文本字符串
        
        Args:
            text: 待切分的文本
            metadata: 可选的元数据字典
            
        Returns:
            切分后的 Document 列表
        """
        doc = Document(
            page_content=text,
            metadata=metadata or {}
        )
        return self.split_documents([doc])




def split_markdown_legacy_exact(
    text: str,
    paper_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Run the original ``small-model/splitter.py`` algorithm and adapt it.

    This is the default document splitter for both LLM and SLM RAG so that
    model-size comparisons use identical candidate chunks.  The v2 structured
    splitter remains available only as an explicit ablation; table-aware strategies can still add
    row-level table evidence on top of these base chunks.
    """
    splitter = CustomMarkdownSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
    )
    docs = splitter.split_text(
        text,
        metadata={"source": paper_id, "paper_id": paper_id},
    )
    chunks: List[Dict[str, Any]] = []
    total = len(docs)
    for i, doc in enumerate(docs):
        meta = dict(getattr(doc, "metadata", {}) or {})
        section_title = (
            meta.get("header_4")
            or meta.get("header_3")
            or meta.get("header_2")
            or meta.get("header_1")
            or meta.get("title")
            or ""
        )
        content_type = str(meta.get("content_type") or "text")
        chunk_id = str(meta.get("chunk_id") or f"{paper_id}#chunk_{i}")
        chunk_index = int(meta.get("chunk_index_in_doc", i))
        total_in_doc = int(meta.get("total_chunks_in_doc", total))
        chunks.append({
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "unit_id": chunk_id,
            "unit_type": "chunk",
            "chunk_type": content_type,
            "content_type": content_type,
            "section": section_title,
            "section_title": section_title,
            "text": str(getattr(doc, "page_content", "") or ""),
            "chunk_index_in_doc": chunk_index,
            "chunk_index": chunk_index,
            "total_chunks_in_doc": total_in_doc,
            "total_chunks": total_in_doc,
            "metadata": meta,
        })
    return chunks
