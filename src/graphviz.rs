use crate::filesystem::*;
use crate::graphviz_concat;

use crate::GraphStructure;
use crate::Layer;

use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::{
    attributes::*,
    cmd::{CommandArg, Format},
    exec, exec_dot, parse,
    printer::{DotPrinter, PrinterContext},
};

use std::fmt;

// Sets general style
fn header_generate_nn_graph_string(layer_spacing: f32) -> String {
    let result_string = format!(
        r##"digraph NNGraph {{
        bgcolor="#222222"
        node [style=filled, fillcolor="#444444", fontcolor="#FFFFFFF", color = "#FFFFFF", shape="circle"]
        edge [fontcolor="#FFFFFFF", color = "#FFFFFF"]
        graph [fontcolor="#FFFFFFF", color = "#FFFFFF"]

        rankdir = LR;
        splines=false;
        edge[style=invis, dir=none];
        ranksep= {};
        {{
            node [shape=circle, color="#ffcc00", style=filled, fillcolor="#ffcc00"];
        }}
        
        "##,
        layer_spacing
    );

    result_string
}

// body code
fn body_input_generate_nn_graph_string(num_input_nodes: i32) -> String {
    let mut string_input_nodes: String = String::new();
    for i in 0..num_input_nodes {
        string_input_nodes += "x";
        string_input_nodes += i.to_string().as_str();
        string_input_nodes += " [label=<x<sub>";
        string_input_nodes += i.to_string().as_str();
        string_input_nodes += " </sub>>];\n"
    }

    let result_string = format!(
        r##"
        {{
            node [shape=circle, color="#00b300", style=filled, fillcolor="#00b300"];
            // 1
            {}
        }}
        "##,
        string_input_nodes
    );

    result_string
}

fn body_hidden_generate_nn_graph_string(num_hidden_nodes: &[i32]) -> String {
    let string_start = r##"
    {
        node [shape=circle, color=dodgerblue, style=filled, fillcolor=dodgerblue];
    "##;

    let mut string_hidden = String::new();
    for i in (0 + 2)..(num_hidden_nodes.len() + 2) {
        // Comment
        string_hidden += "//";
        string_hidden += &i.to_string();
        string_hidden += "\n";

        // each node in layer i
        let mut string_hidden_nodes = String::new();
        for n in (0)..(num_hidden_nodes[i - 2] as usize) {
            string_hidden_nodes += "a";
            string_hidden_nodes += &n.to_string();
            string_hidden_nodes += &i.to_string();
            string_hidden_nodes += " [label=<a<sub>";
            string_hidden_nodes += &n.to_string();
            string_hidden_nodes += "</sub><sup>(";
            string_hidden_nodes += &i.to_string();
            string_hidden_nodes += ")</sup>>];\n"
        }
        string_hidden += &string_hidden_nodes.to_owned();
    }

    // Prevent tilting
    const prevent_tiling: bool = false;
    if (prevent_tiling) {
        string_hidden += "// Prevent tilting\n";
        for i in (0 + 2)..(num_hidden_nodes.len() - 1 + 2) {
            string_hidden += "a0";
            string_hidden += &i.to_string();
            string_hidden += "->";
            string_hidden += "a0";
            string_hidden += &(i + 1).to_string();
            string_hidden += "\n";
        }
    }

    let result_string = format!(
        r##"
    {{
        node [shape=circle, color=dodgerblue, style=filled, fillcolor=dodgerblue];
        {}
    }}
    "##,
        string_hidden
    );

    result_string
}

fn body_output_generate_nn_graph_string(output_layer_index: i32, num_output_nodes: i32) -> String {
    let mut string_output_nodes: String = String::new();
    // Comment
    string_output_nodes += "//";
    string_output_nodes += &output_layer_index.to_string();
    string_output_nodes += "\n";

    // Output layer
    for i in 0..num_output_nodes {
        string_output_nodes += "y";
        string_output_nodes += &i.to_string();
        string_output_nodes += " [label=<y<sub>";
        string_output_nodes += &i.to_string();
        string_output_nodes += "</sub><sup>(";
        string_output_nodes += &output_layer_index.to_string();
        string_output_nodes += ")</sup>>];";
        string_output_nodes += "\n";
    }

    let result_string: String = format!(
        r##"
    {{
        node [shape=circle, color=coral1, style=filled, fillcolor=coral1];

        {}
    }}
    "##,
        string_output_nodes
    );
    result_string
}

fn body_rank_generate_nn_graph_string(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
) -> String {
    let mut string_rank: String = String::new();

    // Input layer
    string_rank += "{\n";
    string_rank += "rank=same;\n";
    for i in 0..num_input_nodes - 1 {
        string_rank += "x";
        string_rank += &i.to_string();
        string_rank += "->";
    }
    // last manual without adding arrow
    string_rank += "x";
    string_rank += &(num_input_nodes - 1).to_string();
    string_rank += ";\n}\n";

    // Hidden layer
    for l in (0 + 2)..(num_hidden_nodes.len() + 2) {
        string_rank += "{\n";
        string_rank += "rank=same;\n";
        for i in 0..num_hidden_nodes[l - 2] - 1 {
            string_rank += "a";
            string_rank += &i.to_string();
            string_rank += &l.to_string();
            string_rank += "->";
        }
        // last manual without adding arrow
        string_rank += "a";
        string_rank += &(num_hidden_nodes[l - 2] - 1).to_string();
        string_rank += &l.to_string();
        string_rank += ";\n}\n";
    }

    // Output layer
    string_rank += "{\n";
    string_rank += "rank=same;\n";
    for i in 0..num_output_nodes - 1 {
        string_rank += "y";
        string_rank += &i.to_string();
        string_rank += "->";
    }
    // last manual without adding arrow
    string_rank += "y";
    string_rank += &(num_output_nodes - 1).to_string();
    string_rank += ";\n}\n";

    string_rank
}

fn body_text_generate_nn_graph_string(num_hidden_nodes: &[i32]) -> String {
    let mut string_layer_labels = String::new();
    for i in (0 + 2)..(num_hidden_nodes.len() + 2) {
        string_layer_labels += "l";
        string_layer_labels += &(i - 1).to_string();
        string_layer_labels += " [shape=plaintext, label=\"Hidden Layer [";
        string_layer_labels += &(i).to_string();
        string_layer_labels += "]\"];\n";
        string_layer_labels += "l";
        string_layer_labels += &(i - 1).to_string();
        string_layer_labels += "->a0";
        string_layer_labels += &i.to_string();
        string_layer_labels += "\n";
        string_layer_labels += "{rank=same; ";
        string_layer_labels += "l";
        string_layer_labels += &(i - 1).to_string();
        string_layer_labels += ";";
        string_layer_labels += "a0";
        string_layer_labels += &i.to_string();
        string_layer_labels += "}";
        string_layer_labels += "\n";
    }

    let mut string_layer_labels_output: String = String::new();
    let l_output = num_hidden_nodes.len() + 1;
    string_layer_labels_output += "l";
    string_layer_labels_output += &l_output.to_string();
    string_layer_labels_output += " [shape=plaintext, label=\"Output Layer [";
    string_layer_labels_output += &(l_output + 1).to_string();
    string_layer_labels_output += "]\"];\n";
    string_layer_labels_output += "l";
    string_layer_labels_output += &l_output.to_string();
    string_layer_labels_output += "->y0\n";
    string_layer_labels_output += "{rank=same; l";
    string_layer_labels_output += &l_output.to_string();
    string_layer_labels_output += ";y0};\n";

    let result_string = format!(
        r##"
    l0 [shape=plaintext, label="Input Layer [1]"];
    l0->x0;
    {{rank=same; l0;x0}};
    {}
    {}
    "##,
        string_layer_labels, string_layer_labels_output
    );
    result_string
}

fn body_arrows_generate_nn_graph_string(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
) -> String {
    // Input
    let mut string_layer_input = String::new();
    {
        string_layer_input += "{";

        for i in 0..(num_input_nodes - 1) {
            string_layer_input += "x";
            string_layer_input += &i.to_string();
            string_layer_input += "; "
        }
        // Last manual
        string_layer_input += "x";
        string_layer_input += &(num_input_nodes - 1).to_string();
        string_layer_input += "}";
    }

    // Hidden
    let mut string_layer_all_hidden = String::new();
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

            string_layer_all_hidden += "->";
            string_layer_all_hidden += &string_layer;
            string_layer_all_hidden += ";\n";
            string_layer_all_hidden += &string_layer;
        }
    }

    // Output
    let mut string_layer_output = String::new();
    {
        string_layer_output += "->";
        string_layer_output += "{";

        for i in 0..(num_output_nodes - 1) {
            string_layer_output += "y";
            string_layer_output += &i.to_string();
            string_layer_output += "; ";
        }
        // Last manual
        string_layer_output += "y";
        string_layer_output += &(num_output_nodes - 1).to_string();
        string_layer_output += "}";
    }

    let result_string: String = format!(
        r##"
    edge[style=solid, tailport=e, headport=w];
    edge[labelangle=0, labelfloat = false, labeldistance = 8];
    {}{}{}
    "##,
        string_layer_input, string_layer_all_hidden, string_layer_output
    );
    result_string
}

fn body_arrow_labels_generate_nn_graph_string(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
    nn_layers: &Vec<Layer>,
) -> String {
    // edge[style=solid, tailport=e, headport=w];

    // edge[labelangle=0, labelfloat = false, labeldistance = 6];
    // edge[color="white;0.2:#222222;0.25:white"]

    // a02->y0[taillabel="0.020"];
    // a12->y0[taillabel="0.021"];
    // a22->y0[taillabel="0.022"];

    // a02->y1[taillabel="0.030"];
    // a12->y1[taillabel="0.031"];
    // a22->y1[taillabel="0.032"];

    let result_string = String::new();
    result_string
}

fn body_generate_nn_graph_layout_string(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
) -> String {
    let output_layer_index: i32 = num_hidden_nodes.len() as i32 + 2;

    let mut body: String = body_input_generate_nn_graph_string(num_input_nodes);
    body.push_str(body_hidden_generate_nn_graph_string(&num_hidden_nodes).as_str());
    body.push_str(
        body_output_generate_nn_graph_string(output_layer_index, num_output_nodes).as_str(),
    );
    body.push_str(
        body_rank_generate_nn_graph_string(num_input_nodes, &num_hidden_nodes, num_output_nodes)
            .as_str(),
    );
    body.push_str(body_text_generate_nn_graph_string(&num_hidden_nodes).as_str());
    body.push_str(
        body_arrows_generate_nn_graph_string(num_input_nodes, &num_hidden_nodes, num_output_nodes)
            .as_str(),
    );

    body
}

fn body_generate_nn_graph_weight_bias_string(
    num_input_nodes: i32,
    num_hidden_nodes: &[i32],
    num_output_nodes: i32,
    nn_layers: &Vec<Layer>,
) -> String {
    let output_layer_index: i32 = num_hidden_nodes.len() as i32 + 2;

    let mut body: String = body_input_generate_nn_graph_string(num_input_nodes);
    body.push_str(body_hidden_generate_nn_graph_string(&num_hidden_nodes).as_str());
    body.push_str(
        body_output_generate_nn_graph_string(output_layer_index, num_output_nodes).as_str(),
    );
    body.push_str(
        body_rank_generate_nn_graph_string(num_input_nodes, &num_hidden_nodes, num_output_nodes)
            .as_str(),
    );
    body.push_str(body_text_generate_nn_graph_string(&num_hidden_nodes).as_str());

    // Add weight & bias information
    body.push_str(
        &body_arrow_labels_generate_nn_graph_string(
            num_input_nodes,
            &num_hidden_nodes,
            num_output_nodes,
            nn_layers,
        )
        .as_str(),
    );

    body
}

// Finishes and close brackets
fn footer_generate_nn_graph_string() -> String {
    let footer: String = "}".to_string();
    footer
}

pub struct GenerateGraphParams {
    pub layer_spacing: f32,
}

pub fn generate_nn_graph_layout_string(
    graph_structure: &GraphStructure,
    params: &GenerateGraphParams,
) -> String {
    // TODO: Fix prettier string formating

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

    result_string.push_str(header_generate_nn_graph_string(params.layer_spacing).as_str());
    result_string.push_str(
        body_generate_nn_graph_layout_string(num_input_nodes, &num_hidden_nodes, num_output_nodes)
            .as_str(),
    );
    result_string.push_str(footer_generate_nn_graph_string().as_str());

    result_string
}

pub fn generate_nn_graph_weight_bias_string(
    graph_structure: &GraphStructure,
    params: &GenerateGraphParams,
    nn_layers: &Vec<Layer>,
) -> String {
    // TODO: Fix prettier string formating

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

    result_string.push_str(header_generate_nn_graph_string(params.layer_spacing).as_str());
    result_string.push_str(
        body_generate_nn_graph_weight_bias_string(
            num_input_nodes,
            &num_hidden_nodes,
            num_output_nodes,
            nn_layers,
        )
        .as_str(),
    );
    result_string.push_str(footer_generate_nn_graph_string().as_str());
    result_string
}
