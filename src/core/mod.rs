//! Core types for symbolic mathematics
//!
//! This module contains the fundamental types:
//! - `Expr` / `ExprKind` - Expression AST
//! - `Symbol` / `InternedSymbol` - Symbol system  
//! - `Polynomial` - Polynomial representation
//! - `DiffError` - Error types
//! - Display formatting (to_string, to_latex, to_unicode)
//! - Visitor pattern for AST traversal

mod display; // Display implementations for Expr
pub(crate) mod error;
pub(crate) mod expr;
pub(crate) mod poly;
pub(crate) mod symbol;
pub(crate) mod traits;
pub mod visitor; // Public visitor pattern

// Public re-exports (for external API)
pub use error::{DiffError, Span};
pub use expr::{Expr, ExprKind};
pub use symbol::{
    InternedSymbol, Symbol, SymbolContext, SymbolError, clear_symbols, global_context,
    remove_symbol, symb, symb_get, symb_new, symbol_count, symbol_exists, symbol_names,
};
