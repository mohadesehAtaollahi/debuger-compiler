#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================
 DEBUG LANGUAGE COMPILER - LR(0) PARSER
===========================================
 Grammar:
   Program    → StmtList
   StmtList   → StmtList Stmt | Stmt
   Stmt       → INT NUMBER
              | BREAK IDENTIFIER
              | WATCH IDENTIFIER  
              | STEP
              | CONTINUE
              | PRINT Expr
   Expr       → Expr + Term | Expr - Term | Term
   Term       → Term * Factor | Term / Factor | Factor
   Factor     → ( Expr ) | NUMBER | IDENTIFIER
===========================================
"""

import json
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import os

# ============================================================
#                        LEXER
# ============================================================

class Token:
    def __init__(self, type_, value, line, col):
        self.type = type_
        self.value = value
        self.line = line
        self.col = col
    
    def to_dict(self):
        return {
            'type': self.type,
            'value': self.value,
            'line': self.line,
            'col': self.col
        }
    
    def __repr__(self):
        return f"Token({self.type}, {self.value})"

class Lexer:
    KEYWORDS = {
        'INT': 'INT', 'int': 'INT',
        'BREAK': 'BREAK', 'break': 'BREAK',
        'WATCH': 'WATCH', 'watch': 'WATCH',
        'STEP': 'STEP', 'step': 'STEP',
        'CONTINUE': 'CONTINUE', 'continue': 'CONTINUE',
        'PRINT': 'PRINT', 'print': 'PRINT'
    }
    
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
    
    def tokenize(self):
        tokens = []
        errors = []
        
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            
            # Whitespace
            if ch in ' \t\r':
                self.pos += 1
                self.col += 1
                continue
            
            # Newline
            if ch == '\n':
                self.pos += 1
                self.line += 1
                self.col = 1
                continue
            
            # Comment
            if ch == '/' and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == '/':
                while self.pos < len(self.text) and self.text[self.pos] != '\n':
                    self.pos += 1
                continue
            
            # Number
            if ch.isdigit():
                start = self.pos
                start_col = self.col
                while self.pos < len(self.text) and self.text[self.pos].isdigit():
                    self.pos += 1
                    self.col += 1
                tokens.append(Token('NUMBER', self.text[start:self.pos], self.line, start_col))
                continue
            
            # Identifier / Keyword
            if ch.isalpha() or ch == '_':
                start = self.pos
                start_col = self.col
                while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
                    self.pos += 1
                    self.col += 1
                word = self.text[start:self.pos]
                token_type = self.KEYWORDS.get(word, 'IDENTIFIER')
                tokens.append(Token(token_type, word, self.line, start_col))
                continue
            
            # Operators
            if ch == '+':
                tokens.append(Token('PLUS', '+', self.line, self.col))
            elif ch == '-':
                tokens.append(Token('MINUS', '-', self.line, self.col))
            elif ch == '*':
                tokens.append(Token('MULTIPLY', '*', self.line, self.col))
            elif ch == '/':
                tokens.append(Token('DIVIDE', '/', self.line, self.col))
            elif ch == '(':
                tokens.append(Token('LPAREN', '(', self.line, self.col))
            elif ch == ')':
                tokens.append(Token('RPAREN', ')', self.line, self.col))
            elif ch == ';':
                tokens.append(Token('SEMICOLON', ';', self.line, self.col))
            elif ch == '=':
                tokens.append(Token('ASSIGN', '=', self.line, self.col))
            else:
                errors.append(f"Unknown character '{ch}' at line {self.line}, col {self.col}")
            
            self.pos += 1
            self.col += 1
        
        tokens.append(Token('$', '$', self.line, self.col))
        return tokens, errors


# ============================================================
#                        GRAMMAR
# ============================================================

class Production:
    def __init__(self, id_, left, right):
        self.id = id_
        self.left = left
        self.right = right  # List of symbols
    
    def __str__(self):
        right_str = ' '.join(self.right) if self.right else 'ε'
        return f"{self.left} → {right_str}"
    
    def __repr__(self):
        return f"P{self.id}: {self}"

class Grammar:
    def __init__(self):
        self.productions = []
        self.terminals = set()
        self.non_terminals = set()
        self.start_symbol = "S'"
        self.first = {}
        self.follow = {}
        
        self._build()
        self._compute_first()
        self._compute_follow()
    
    def _build(self):
        """Build the debug language grammar"""
        rules = [
            ("S'", ["Program"]),                           # 0: Augmented
            ("Program", ["StmtList"]),                     # 1
            ("StmtList", ["StmtList", "Stmt"]),           # 2
            ("StmtList", ["Stmt"]),                        # 3
            ("Stmt", ["IntStmt"]),                         # 4
            ("Stmt", ["BreakStmt"]),                       # 5
            ("Stmt", ["WatchStmt"]),                       # 6
            ("Stmt", ["StepStmt"]),                        # 7
            ("Stmt", ["ContStmt"]),                        # 8
            ("Stmt", ["PrintStmt"]),                       # 9
            ("IntStmt", ["INT", "NUMBER"]),               # 10
            ("BreakStmt", ["BREAK", "IDENTIFIER"]),       # 11
            ("WatchStmt", ["WATCH", "IDENTIFIER"]),       # 12
            ("StepStmt", ["STEP"]),                        # 13
            ("ContStmt", ["CONTINUE"]),                    # 14
            ("PrintStmt", ["PRINT", "Expr"]),             # 15
            ("Expr", ["Expr", "PLUS", "Term"]),           # 16
            ("Expr", ["Expr", "MINUS", "Term"]),          # 17
            ("Expr", ["Term"]),                            # 18
            ("Term", ["Term", "MULTIPLY", "Factor"]),     # 19
            ("Term", ["Term", "DIVIDE", "Factor"]),       # 20
            ("Term", ["Factor"]),                          # 21
            ("Factor", ["LPAREN", "Expr", "RPAREN"]),     # 22
            ("Factor", ["NUMBER"]),                        # 23
            ("Factor", ["IDENTIFIER"]),                    # 24
        ]
        
        # Terminals
        self.terminals = {
            'INT', 'BREAK', 'WATCH', 'STEP', 'CONTINUE', 'PRINT',
            'NUMBER', 'IDENTIFIER',
            'PLUS', 'MINUS', 'MULTIPLY', 'DIVIDE',
            'LPAREN', 'RPAREN', 'SEMICOLON',
            '$'
        }
        
        for i, (left, right) in enumerate(rules):
            self.productions.append(Production(i, left, right))
            self.non_terminals.add(left)
    
    def _compute_first(self):
        """Compute FIRST sets"""
        # Initialize
        for t in self.terminals:
            self.first[t] = {t}
        for nt in self.non_terminals:
            self.first[nt] = set()
        
        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                before = len(self.first[prod.left])
                
                if not prod.right:
                    self.first[prod.left].add('ε')
                else:
                    for symbol in prod.right:
                        if symbol in self.first:
                            self.first[prod.left].update(self.first[symbol] - {'ε'})
                            if 'ε' not in self.first.get(symbol, set()):
                                break
                    else:
                        self.first[prod.left].add('ε')
                
                if len(self.first[prod.left]) > before:
                    changed = True
    
    def _compute_follow(self):
        """Compute FOLLOW sets"""
        for nt in self.non_terminals:
            self.follow[nt] = set()
        
        self.follow["S'"].add('$')
        
        changed = True
        while changed:
            changed = False
            for prod in self.productions:
                for i, symbol in enumerate(prod.right):
                    if symbol in self.non_terminals:
                        before = len(self.follow[symbol])
                        rest = prod.right[i + 1:]
                        
                        if rest:
                            first_rest = set()
                            for s in rest:
                                first_rest.update(self.first.get(s, set()) - {'ε'})
                                if 'ε' not in self.first.get(s, set()):
                                    break
                            else:
                                first_rest.add('ε')
                            
                            self.follow[symbol].update(first_rest - {'ε'})
                            if 'ε' in first_rest:
                                self.follow[symbol].update(self.follow[prod.left])
                        else:
                            self.follow[symbol].update(self.follow[prod.left])
                        
                        if len(self.follow[symbol]) > before:
                            changed = True
    
    def to_dict(self):
        return {
            'productions': [str(p) for p in self.productions],
            'terminals': sorted(list(self.terminals)),
            'non_terminals': sorted(list(self.non_terminals)),
            'first': {k: sorted(list(v)) for k, v in self.first.items() if k in self.non_terminals},
            'follow': {k: sorted(list(v)) for k, v in self.follow.items()}
        }


# ============================================================
#                     LR(0) PARSER
# ============================================================

class LR0Item:
    def __init__(self, prod_id, dot_pos, production):
        self.prod_id = prod_id
        self.dot_pos = dot_pos
        self.production = production
    
    def __eq__(self, other):
        return self.prod_id == other.prod_id and self.dot_pos == other.dot_pos
    
    def __hash__(self):
        return hash((self.prod_id, self.dot_pos))
    
    def __str__(self):
        right = list(self.production.right)
        right.insert(self.dot_pos, '•')
        return f"[{self.production.left} → {' '.join(right)}]"
    
    def next_symbol(self):
        if self.dot_pos < len(self.production.right):
            return self.production.right[self.dot_pos]
        return None
    
    def is_complete(self):
        return self.dot_pos >= len(self.production.right)
    
    def advance(self):
        return LR0Item(self.prod_id, self.dot_pos + 1, self.production)

class LR0State:
    def __init__(self, id_, items):
        self.id = id_
        self.items = frozenset(items)
        self.transitions = {}
    
    def __eq__(self, other):
        return self.items == other.items
    
    def __hash__(self):
        return hash(self.items)

class LR0Parser:
    def __init__(self, grammar):
        self.grammar = grammar
        self.states = []
        self.action = {}
        self.goto = {}
        self.conflicts = []
        
        self._build_automaton()
        self._build_tables()
    
    def _closure(self, items):
        """Compute closure of item set"""
        closure = set(items)
        changed = True
        
        while changed:
            changed = False
            new_items = set()
            
            for item in closure:
                next_sym = item.next_symbol()
                if next_sym and next_sym in self.grammar.non_terminals:
                    for prod in self.grammar.productions:
                        if prod.left == next_sym:
                            new_item = LR0Item(prod.id, 0, prod)
                            if new_item not in closure:
                                new_items.add(new_item)
                                changed = True
            
            closure.update(new_items)
        
        return closure
    
    def _goto(self, items, symbol):
        """Compute GOTO(items, symbol)"""
        moved = set()
        for item in items:
            if item.next_symbol() == symbol:
                moved.add(item.advance())
        
        if moved:
            return self._closure(moved)
        return set()
    
    def _build_automaton(self):
        """Build LR(0) automaton"""
        # Initial state
        start_prod = self.grammar.productions[0]
        start_item = LR0Item(0, 0, start_prod)
        start_items = self._closure({start_item})
        start_state = LR0State(0, start_items)
        
        self.states = [start_state]
        state_map = {frozenset(start_items): 0}
        worklist = [start_state]
        
        while worklist:
            state = worklist.pop(0)
            
            # Find all symbols after dots
            symbols = set()
            for item in state.items:
                sym = item.next_symbol()
                if sym:
                    symbols.add(sym)
            
            for symbol in symbols:
                goto_items = self._goto(state.items, symbol)
                if goto_items:
                    frozen = frozenset(goto_items)
                    if frozen in state_map:
                        state.transitions[symbol] = state_map[frozen]
                    else:
                        new_id = len(self.states)
                        new_state = LR0State(new_id, goto_items)
                        self.states.append(new_state)
                        state_map[frozen] = new_id
                        state.transitions[symbol] = new_id
                        worklist.append(new_state)
    
    def _build_tables(self):
        """Build ACTION and GOTO tables"""
        for state in self.states:
            for item in state.items:
                if item.is_complete():
                    if item.prod_id == 0:
                        # Accept
                        self._add_action(state.id, '$', ('accept', None))
                    else:
                        # Reduce
                        for terminal in self.grammar.follow.get(item.production.left, set()):
                            self._add_action(state.id, terminal, ('reduce', item.prod_id))
                else:
                    next_sym = item.next_symbol()
                    if next_sym in self.grammar.terminals:
                        if next_sym in state.transitions:
                            self._add_action(state.id, next_sym, 
                                           ('shift', state.transitions[next_sym]))
            
            # GOTO for non-terminals
            for symbol, next_id in state.transitions.items():
                if symbol in self.grammar.non_terminals:
                    self.goto[(state.id, symbol)] = next_id
    
    def _add_action(self, state_id, symbol, action):
        key = (state_id, symbol)
        if key in self.action:
            if self.action[key] != action:
                self.conflicts.append({
                    'state': state_id,
                    'symbol': symbol,
                    'actions': [self._format_action(self.action[key]), 
                               self._format_action(action)]
                })
        self.action[key] = action
    
    def _format_action(self, action):
        type_, value = action
        if type_ == 'shift':
            return f"s{value}"
        elif type_ == 'reduce':
            return f"r{value}"
        elif type_ == 'accept':
            return "acc"
        return ""
    
    def parse(self, tokens):
        """Parse tokens and return steps + parse tree"""
        steps = []
        stack = [0]
        symbols = []
        nodes = []
        pos = 0
        step_num = 0
        
        while True:
            state = stack[-1]
            token = tokens[pos] if pos < len(tokens) else tokens[-1]
            lookahead = token.type
            
            action = self.action.get((state, lookahead))
            
            # Record step
            step = {
                'step': step_num,
                'stack': list(stack),
                'symbols': list(symbols),
                'input': [t.value for t in tokens[pos:]],
                'action': '',
                'production': ''
            }
            
            if action is None:
                step['action'] = f"Error: unexpected '{lookahead}'"
                steps.append(step)
                return {
                    'success': False,
                    'steps': steps,
                    'tree': None,
                    'error': f"Syntax error: unexpected '{token.value}' at line {token.line}"
                }
            
            type_, value = action
            
            if type_ == 'shift':
                step['action'] = f"Shift {value}"
                stack.append(value)
                symbols.append(lookahead)
                nodes.append({
                    'symbol': lookahead,
                    'value': token.value,
                    'children': []
                })
                pos += 1
            
            elif type_ == 'reduce':
                prod = self.grammar.productions[value]
                step['action'] = f"Reduce {value}"
                step['production'] = str(prod)
                
                children = []
                for _ in range(len(prod.right)):
                    stack.pop()
                    symbols.pop()
                    if nodes:
                        children.insert(0, nodes.pop())
                
                # Create parent node
                parent = {
                    'symbol': prod.left,
                    'value': prod.left,
                    'children': children
                }
                nodes.append(parent)
                
                # GOTO
                top = stack[-1]
                goto_state = self.goto.get((top, prod.left))
                if goto_state is None:
                    step['action'] = "Error: no GOTO"
                    steps.append(step)
                    return {
                        'success': False,
                        'steps': steps,
                        'tree': None,
                        'error': "Internal error: no GOTO state"
                    }
                
                stack.append(goto_state)
                symbols.append(prod.left)
            
            elif type_ == 'accept':
                step['action'] = "Accept"
                steps.append(step)
                return {
                    'success': True,
                    'steps': steps,
                    'tree': nodes[0] if nodes else None,
                    'error': None
                }
            
            steps.append(step)
            step_num += 1
            
            if step_num > 1000:
                return {
                    'success': False,
                    'steps': steps,
                    'tree': None,
                    'error': "Parsing took too long"
                }
    
    def get_states_info(self):
        """Get states information"""
        result = []
        for state in self.states:
            result.append({
                'id': state.id,
                'items': [str(item) for item in sorted(state.items, key=str)],
                'transitions': {sym: sid for sym, sid in state.transitions.items()}
            })
        return result
    
    def get_tables(self):
        """Get ACTION and GOTO tables"""
        # Terminals for ACTION
        terminals = sorted(list(self.grammar.terminals))
        
        # Non-terminals for GOTO (excluding S')
        non_terminals = sorted([nt for nt in self.grammar.non_terminals if nt != "S'"])
        
        action_table = []
        goto_table = []
        
        for state_id in range(len(self.states)):
            action_row = {}
            for t in terminals:
                act = self.action.get((state_id, t))
                if act:
                    action_row[t] = self._format_action(act)
                else:
                    action_row[t] = ''
            action_table.append(action_row)
            
            goto_row = {}
            for nt in non_terminals:
                g = self.goto.get((state_id, nt))
                goto_row[nt] = str(g) if g is not None else ''
            goto_table.append(goto_row)
        
        return {
            'terminals': terminals,
            'non_terminals': non_terminals,
            'action': action_table,
            'goto': goto_table,
            'conflicts': self.conflicts
        }


# ============================================================
#                     HTTP SERVER
# ============================================================

# Global instances
grammar = Grammar()
parser = LR0Parser(grammar)

class CompilerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/':
            # Serve index.html
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            # Read and serve HTML file
            html_path = os.path.join(os.path.dirname(__file__), 'index.html')
            if os.path.exists(html_path):
                with open(html_path, 'r', encoding='utf-8') as f:
                    self.wfile.write(f.read().encode('utf-8'))
            else:
                self.wfile.write(b"<h1>index.html not found</h1>")
            return
        
        elif parsed.path == '/api/grammar':
            self._send_json(grammar.to_dict())
            return
        
        elif parsed.path == '/api/tables':
            self._send_json(parser.get_tables())
            return
        
        elif parsed.path == '/api/states':
            self._send_json(parser.get_states_info())
            return
        
        super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/tokenize':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            lexer = Lexer(data.get('code', ''))
            tokens, errors = lexer.tokenize()
            
            self._send_json({
                'tokens': [t.to_dict() for t in tokens],
                'errors': errors
            })
            return
        
        elif self.path == '/api/parse':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            lexer = Lexer(data.get('code', ''))
            tokens, errors = lexer.tokenize()
            
            if errors:
                self._send_json({
                    'success': False,
                    'error': errors[0],
                    'tokens': [t.to_dict() for t in tokens],
                    'steps': [],
                    'tree': None
                })
                return
            
            result = parser.parse(tokens)
            result['tokens'] = [t.to_dict() for t in tokens]
            self._send_json(result)
            return
        
        self.send_error(404)
    
    def _send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def log_message(self, format, *args):
        print(f"[SERVER] {args[0]}")


def main():
    port = 8080
    server = HTTPServer(('localhost', port), CompilerHandler)
    
    print("=" * 60)
    print("  🔧 DEBUG LANGUAGE COMPILER - LR(0) PARSER")
    print("=" * 60)
    print(f"\n  Grammar: {len(grammar.productions)} productions")
    print(f"  States:  {len(parser.states)} LR(0) states")
    print(f"  Conflicts: {len(parser.conflicts)}")
    print(f"\n  🌐 Open in browser: http://localhost:{port}")
    print("\n  Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n  Server stopped.")
        server.shutdown()

if __name__ == '__main__':
    main()