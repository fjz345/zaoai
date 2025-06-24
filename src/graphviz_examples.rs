use std::fs::File;
use std::io::prelude::*;

use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::{
    attributes::*,
    cmd::{CommandArg, Format},
    exec, exec_dot, parse,
    printer::{DotPrinter, PrinterContext},
};

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}

pub fn parse_test() {
    let g: Graph = parse(
        r#"
        strict digraph t {
            bgcolor="lightblue"
            color="lightblue"
            aa[color=green]
            subgraph v {
                aa[shape=square]
                subgraph vv{a2 -> b2}
                aaa[color=red]
                aaa -> bbb
            }
            aa -> be -> subgraph v { d -> aaa}
            aa -> aaa -> v
        }
        "#,
    )
    .unwrap();

    assert_eq!(
        g,
        graph!(strict di id!("t");
          node!("aa";attr!("color","green")),
          subgraph!("v";
            node!("aa"; attr!("shape","square")),
            subgraph!("vv"; edge!(node_id!("a2") => node_id!("b2"))),
            node!("aaa";attr!("color","red")),
            edge!(node_id!("aaa") => node_id!("bbb"))
            ),
          edge!(node_id!("aa") => node_id!("be") => subgraph!("v"; edge!(node_id!("d") => node_id!("aaa")))),
          edge!(node_id!("aa") => node_id!("aaa") => node_id!("v"))
        )
    )
}

pub fn print_test() {
    let mut g = graph!(strict di id!("id"));
    assert_eq!(
        "strict digraph id {\n\n}".to_string(),
        g.print(&mut PrinterContext::default())
    );
}

pub fn output_test() {
    let mut g = graph!(id!("id");
         node!("nod"),
         subgraph!("sb";
             edge!(node_id!("a") => subgraph!(;
                node!("n";
                NodeAttributes::color(color_name::black), NodeAttributes::shape(shape::egg))
            ))
        ),
        edge!(node_id!("a1") => node_id!(esc "a2"))
    );
    let graph_svg = exec(g, &mut PrinterContext::default(), vec![Format::Svg.into()]).unwrap();
}

pub fn output_exec_from_test() {
    let mut g = graph!(id!("id");
         node!("nod"),
         subgraph!("sb";
             edge!(node_id!("a") => subgraph!(;
                node!("n";
                NodeAttributes::color(color_name::black), NodeAttributes::shape(shape::egg))
            ))
        ),
        edge!(node_id!("a1") => node_id!(esc "a2"))
    );
    let dot = g.print(&mut PrinterContext::default());
    log::info!("{}", dot);
    let format = Format::Svg;

    let graph_svg = exec_dot(dot.clone(), vec![format.into()]).unwrap();

    let graph_svg = exec_dot(dot, vec![format.clone().into()]).unwrap();
}

pub fn graph_test() {
    let g = r#"
            strict digraph t {
                aa[color=green]
                subgraph v {
                    aa[shape=square]
                    subgraph vv{a2 -> b2}
                    aaa[color=red]
                    aaa -> bbb
                }
                aa -> be -> subgraph v { d -> aaa}
                aa -> aaa -> v
            }
            "#;

    let mut file = File::create("test.dot").expect("Failed to write to file");
    file.write_all(unsafe { g.as_bytes() });
}
