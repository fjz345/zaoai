use crate::filesystem::*;

use graphviz_rust::dot_generator::*;
use graphviz_rust::dot_structures::*;
use graphviz_rust::{
    attributes::*,
    cmd::{CommandArg, Format},
    exec, exec_dot, parse,
    printer::{DotPrinter, PrinterContext},
};

// Sets general style
fn header_generate_nn_graph_string() -> String
{
    let header: &str = r##"
    digraph NNGraph {
        bgcolor="#222222"
        node [style=filled, fillcolor="#444444", fontcolor="#FFFFFFF", color = "#FFFFFF", shape="circle"]
        edge [fontcolor="#FFFFFFF", color = "#FFFFFF"]
        graph [fontcolor="#FFFFFFF", color = "#FFFFFF"]
    
        rankdir = LR;
        splines=false;
        edge[style=invis];
        ranksep= 2.2;
        {
            node [shape=circle, color="#ffcc00", style=filled, fillcolor="#ffcc00"];
        }

        "##;
        header.to_string()
}

// body code
fn body_input_generate_nn_graph_string() -> String
{
    let string: &str = r##"
    {
        node [shape=circle, color="#00b300", style=filled, fillcolor="#00b300"];
        // 1
        x0 [label=<x<sub>0</sub>>];
        x1 [label=<x<sub>1</sub>>];
        x2 [label=<x<sub>2</sub>>]; 
        x3 [label=<x<sub>3</sub>>];
    }
    "##;
        string.to_string()
}

fn body_hidden_generate_nn_graph_string() -> String
{
    let string: &str = r##"
    {
        node [shape=circle, color=dodgerblue, style=filled, fillcolor=dodgerblue];
        // 2
        a02 [label=<a<sub>0</sub><sup>(2)</sup>>];
        a12 [label=<a<sub>1</sub><sup>(2)</sup>>];
        a22 [label=<a<sub>2</sub><sup>(2)</sup>>];
        a32 [label=<a<sub>3</sub><sup>(2)</sup>>];
        a42 [label=<a<sub>4</sub><sup>(2)</sup>>];
        a52 [label=<a<sub>5</sub><sup>(2)</sup>>];
        // 3
        a03 [label=<a<sub>0</sub><sup>(3)</sup>>];
        a13 [label=<a<sub>1</sub><sup>(3)</sup>>];
        a23 [label=<a<sub>2</sub><sup>(3)</sup>>];
        a33 [label=<a<sub>3</sub><sup>(3)</sup>>];
        a43 [label=<a<sub>4</sub><sup>(3)</sup>>];
        a53 [label=<a<sub>5</sub><sup>(3)</sup>>];
    }
    a02->a03;  // prevent tilting
    "##;
        string.to_string()
}

fn body_output_generate_nn_graph_string() -> String
{
    let string: &str = r##"
    {
        node [shape=circle, color=coral1, style=filled, fillcolor=coral1];
        // 4
        y1 [label=<y<sub>1</sub><sup>(4)</sup>>];
        y2 [label=<y<sub>2</sub><sup>(4)</sup>>]; 
        y3 [label=<y<sub>3</sub><sup>(4)</sup>>]; 
        y4 [label=<y<sub>4</sub><sup>(4)</sup>>];
    }
    "##;
        string.to_string()
}

fn body_rank_generate_nn_graph_string() -> String
{
    let string: &str = r##"
    {
        rank=same;
        x0->x1->x2->x3;
    }
    {
        rank=same;
        a02->a12->a22->a32->a42->a52;
    }
    {
        rank=same;
        a03->a13->a23->a33->a43->a53;
    }
    {
        rank=same;
        y1->y2->y3->y4;
    }
    "##;
        string.to_string()
}

fn body_text_generate_nn_graph_string() -> String
{
    let string: &str = r##"
    l0 [shape=plaintext, label="Input Layer [1]"];
    l0->x0;
    {rank=same; l0;x0};
    l1 [shape=plaintext, label="Hidden Layer [2]"];
    l1->a02;
    {rank=same; l1;a02};
    l2 [shape=plaintext, label="Hidden Layer [3]"];
    l2->a03;
    {rank=same; l2;a03};
    l3 [shape=plaintext, label="Output Layer [4]"];
    l3->y1;
    {rank=same; l3;y1};
    "##;
        string.to_string()
}
fn body_arrows_generate_nn_graph_string() -> String
{
    let string: &str = r##"
    edge[style=solid, tailport=e, headport=w];
    {x0; x1; x2; x3} -> {a02;a12;a22;a32;a42;a52};
    {a02;a12;a22;a32;a42;a52} -> {a03;a13;a23;a33;a43;a53};
    {a03;a13;a23;a33;a43;a53} -> {y1,y2,y3,y4};
    "##;
        string.to_string()
}

fn body_generate_nn_graph_string() -> String
{
    let mut body: String = body_input_generate_nn_graph_string();
    body.push_str(body_hidden_generate_nn_graph_string().as_str());
    body.push_str(body_output_generate_nn_graph_string().as_str());
    body.push_str(body_rank_generate_nn_graph_string().as_str());
    body.push_str(body_text_generate_nn_graph_string().as_str());
    body.push_str(body_arrows_generate_nn_graph_string().as_str());
    
    body
}

// Finishes and close brackets
fn footer_generate_nn_graph_string() -> String
{
    let footer: String = "\n}".to_string();
    footer
}

pub fn generate_nn_graph_string() -> String
{
    let mut result_string: String = String::new();
    // result_string.reserve();

    result_string.push_str(header_generate_nn_graph_string().as_str());
    result_string.push_str(body_generate_nn_graph_string().as_str());
    result_string.push_str(footer_generate_nn_graph_string().as_str());

    result_string
}