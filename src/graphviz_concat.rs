use crate::filesystem::*;
use crate::GraphStructure;

use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::{
    attributes::*,
    cmd::{CommandArg, Format},
    exec, exec_dot, parse,
    printer::{DotPrinter, PrinterContext},
};

use std::fmt;

/*
This file has some functions to generate graphviz strings
*/

// Sets general style
fn header_generate_nn_graph_string_concat(layer_spacing: f32) -> String {
    let header0: &str = r##"
    digraph NNGraph {
        bgcolor="#222222"
        node [style=filled, fillcolor="#444444", fontcolor="#FFFFFFF", color = "#FFFFFF", shape="circle"]
        edge [fontcolor="#FFFFFFF", color = "#FFFFFF"]
        graph [fontcolor="#FFFFFFF", color = "#FFFFFF"]
    
        rankdir = LR;
        splines=false;
        edge[style=invis];
        ranksep=
        "##;

    let header1 = format!("{}", layer_spacing);

    let header2: &str = r##"
        ;
        {
            node [shape=circle, color="#ffcc00", style=filled, fillcolor="#ffcc00"];
        }

        "##;

    let return_string: String = header0.to_owned() + &header1 + header2;
    return_string
}

// body code
fn body_input_generate_nn_graph_string_concat(num_input_nodes: i32) -> String {
    let string_start: &str = r##"
    {
        node [shape=circle, color="#00b300", style=filled, fillcolor="#00b300"];
        // 1
    "##;

    let mut string_middle: String = String::new();
    for i in 0..num_input_nodes {
        string_middle += "x";
        string_middle += i.to_string().as_str();
        string_middle += " [label=<x<sub>";
        string_middle += i.to_string().as_str();
        string_middle += " </sub>>];\n"
    }

    let string_end: &str = r##"
    }
    "##;

    let result_string: String = string_start.to_owned() + &string_middle + string_end;

    result_string
}

fn body_hidden_generate_nn_graph_string_concat(num_hidden_nodes: &[i32]) -> String {
    let string_start = r##"
    {
        node [shape=circle, color=dodgerblue, style=filled, fillcolor=dodgerblue];
    "##;

    let mut string_middle = String::new();

    for i in (0 + 2)..(num_hidden_nodes.len() + 2) {
        // Comment
        string_middle += "//";
        string_middle += &i.to_string();
        string_middle += "\n";

        // each node in layer i
        for n in (0)..(num_hidden_nodes[i - 2] as usize) {
            string_middle += "a";
            string_middle += &n.to_string();
            string_middle += &i.to_string();
            string_middle += " [label=<a<sub>";
            string_middle += &n.to_string();
            string_middle += "</sub><sup>(";
            string_middle += &i.to_string();
            string_middle += ")</sup>>];\n"
        }
    }

    // Prevent tilting
    const PREVENT_TILING: bool = false;
    if (PREVENT_TILING) {
        string_middle += "// Prevent tilting\n";
        for i in (0 + 2)..(num_hidden_nodes.len() - 1 + 2) {
            string_middle += "a0";
            string_middle += &i.to_string();
            string_middle += "->";
            string_middle += "a0";
            string_middle += &(i + 1).to_string();
            string_middle += "\n";
        }
    }
    let string_end = r##"
    }
    "##;

    let result_string: String = string_start.to_owned() + &string_middle + string_end;
    result_string
}

fn body_output_generate_nn_graph_string_concat(
    output_layer_index: i32,
    num_output_nodes: i32,
) -> String {
    let string_start: &str = r##"
    {
        node [shape=circle, color=coral1, style=filled, fillcolor=coral1];
    "##;

    let mut string_middle: String = String::new();

    // Comment
    string_middle += "//";
    string_middle += &output_layer_index.to_string();
    string_middle += "\n";

    // Output layer
    for i in 0..num_output_nodes {
        string_middle += "y";
        string_middle += &i.to_string();
        string_middle += " [label=<y<sub>";
        string_middle += &i.to_string();
        string_middle += "</sub><sup>(";
        string_middle += &output_layer_index.to_string();
        string_middle += ")</sup>>];";
        string_middle += "\n";
    }

    let string_end: &str = r##"
    }
    "##;

    let result_string: String = string_start.to_owned() + &string_middle + string_end;
    result_string
}

fn body_rank_generate_nn_graph_string_concat(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
) -> String {
    let mut string_middle: String = String::new();

    // Input layer
    string_middle += "{\n";
    string_middle += "rank=same;\n";
    for i in 0..num_input_nodes - 1 {
        string_middle += "x";
        string_middle += &i.to_string();
        string_middle += "->";
    }
    // last manual without adding arrow
    string_middle += "x";
    string_middle += &(num_input_nodes - 1).to_string();
    string_middle += ";\n}\n";

    // Hidden layer
    for l in (0 + 2)..(num_hidden_nodes.len() + 2) {
        string_middle += "{\n";
        string_middle += "rank=same;\n";
        for i in 0..num_hidden_nodes[l - 2] - 1 {
            string_middle += "a";
            string_middle += &i.to_string();
            string_middle += &l.to_string();
            string_middle += "->";
        }
        // last manual without adding arrow
        string_middle += "a";
        string_middle += &(num_hidden_nodes[l - 2] - 1).to_string();
        string_middle += &l.to_string();
        string_middle += ";\n}\n";
    }

    // Output layer
    string_middle += "{\n";
    string_middle += "rank=same;\n";
    for i in 0..num_output_nodes - 1 {
        string_middle += "y";
        string_middle += &i.to_string();
        string_middle += "->";
    }
    // last manual without adding arrow
    string_middle += "y";
    string_middle += &(num_output_nodes - 1).to_string();
    string_middle += ";\n}\n";

    string_middle
}

fn body_text_generate_nn_graph_string_concat(num_hidden_nodes: &[i32]) -> String {
    let string_l0_start = r##"
    l0 [shape=plaintext, label="Input Layer [1]"];
    l0->x0;
    {rank=same; l0;x0};
    "##;

    let mut string_middle = String::new();

    for i in (0 + 2)..(num_hidden_nodes.len() + 2) {
        string_middle += "l";
        string_middle += &(i - 1).to_string();
        string_middle += " [shape=plaintext, label=\"Hidden Layer [";
        string_middle += &(i).to_string();
        string_middle += "]\"];\n";
        string_middle += "l";
        string_middle += &(i - 1).to_string();
        string_middle += "->a0";
        string_middle += &i.to_string();
        string_middle += "\n";
        string_middle += "{rank=same; ";
        string_middle += "l";
        string_middle += &(i - 1).to_string();
        string_middle += ";";
        string_middle += "a0";
        string_middle += &i.to_string();
        string_middle += "}";
        string_middle += "\n";
    }

    let mut string_end: String = String::new();
    let l_output = num_hidden_nodes.len() + 1;
    string_end += "l";
    string_end += &l_output.to_string();
    string_end += " [shape=plaintext, label=\"Output Layer [";
    string_end += &(l_output + 1).to_string();
    string_end += "]\"];\n";
    string_end += "l";
    string_end += &l_output.to_string();
    string_end += "->y0\n";
    string_end += "{rank=same; l";
    string_end += &l_output.to_string();
    string_end += ";y0};\n";

    let result_string = string_l0_start.to_owned() + &string_middle + &string_end;
    result_string
}
fn body_arrows_generate_nn_graph_string_concat(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
) -> String {
    let mut string_start: String = String::new();
    string_start += "edge[style=solid, tailport=e, headport=w];\n";

    // Input
    {
        string_start += "{";

        for i in 0..(num_input_nodes - 1) {
            string_start += "x";
            string_start += &i.to_string();
            string_start += "; "
        }
        // Last manual
        string_start += "x";
        string_start += &(num_input_nodes - 1).to_string();
        string_start += "}";
    }

    // Hidden
    {
        for l in (0 + 2)..(num_hidden_nodes.len() + 2) {
            // Create layer {} thingy
            let mut string_layer: String = String::new();
            {
                string_layer += "{";
                for i in 0..num_hidden_nodes[l - 2] - 1 {
                    string_layer += "a";
                    string_layer += &i.to_string();
                    string_layer += &l.to_string();
                    string_layer += ";"
                }
                // Last manual
                string_layer += "a";
                string_layer += &(num_hidden_nodes[l - 2] - 1).to_string();
                string_layer += &l.to_string();
                string_layer += "}";
            }

            string_start += "->";
            string_start += &string_layer;
            string_start += ";\n";
            string_start += &string_layer;
        }
    }

    // Output
    {
        string_start += "->";
        string_start += "{";

        for i in 0..(num_output_nodes - 1) {
            string_start += "y";
            string_start += &i.to_string();
            string_start += ", ";
        }
        // Last manual
        string_start += "y";
        string_start += &(num_output_nodes - 1).to_string();
        string_start += "}";
    }

    let result_string: String = string_start.to_owned();
    result_string
}

fn body_generate_nn_graph_string_concat(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
) -> String {
    let output_layer_index: i32 = num_hidden_nodes.len() as i32 + 2;

    let mut body: String = body_input_generate_nn_graph_string_concat(num_input_nodes);
    body.push_str(body_hidden_generate_nn_graph_string_concat(&num_hidden_nodes).as_str());
    body.push_str(
        body_output_generate_nn_graph_string_concat(output_layer_index, num_output_nodes).as_str(),
    );
    body.push_str(
        body_rank_generate_nn_graph_string_concat(
            num_input_nodes,
            &num_hidden_nodes,
            num_output_nodes,
        )
        .as_str(),
    );
    body.push_str(body_text_generate_nn_graph_string_concat(&num_hidden_nodes).as_str());
    body.push_str(
        body_arrows_generate_nn_graph_string_concat(
            num_input_nodes,
            &num_hidden_nodes,
            num_output_nodes,
        )
        .as_str(),
    );

    body
}

// Finishes and close brackets
fn footer_generate_nn_graph_string_concat() -> String {
    let footer: String = "\n}".to_string();
    footer
}

pub fn generate_nn_graph_string_concat(graph_structure: &GraphStructure) -> String {
    // TODO: Fix prettier string formating
    // TODO: Use some kind of string format which did not work with r##""##

    let mut result_string: String = String::new();
    // result_string.reserve();
    let num_input_nodes = graph_structure.input_nodes as i32;
    // Lazy convert
    let num_hidden_nodes: Vec<i32> = graph_structure
        .hidden_layers
        .iter()
        .map(|&e| e as i32)
        .collect();
    let num_output_nodes = graph_structure.output_nodes as i32;

    result_string.push_str(header_generate_nn_graph_string_concat(2.2).as_str());
    result_string.push_str(
        body_generate_nn_graph_string_concat(num_input_nodes, &num_hidden_nodes, num_output_nodes)
            .as_str(),
    );
    result_string.push_str(footer_generate_nn_graph_string_concat().as_str());

    result_string
}
