use std::{
    collections::{HashMap, VecDeque},
    env,
    fmt::Debug,
    io::Read,
};

use thiserror::Error;

#[derive(Debug)]
struct HuffmanTree {
    //odd index-> leaf node
    //odd index-> internal node except for the last one
    nodes: Vec<Node>,
}

// #[derive(Debug)]
// struct Node {
//     kind: TreeNodeType,
//     frequency: usize,
//     left: usize,
//     right: usize,
// }

// #[derive(Debug)]
// struct Leaf {
//     symbol: char,
//     frequency: usize,
// }

// #[derive(Debug)]
// struct Internal {
//     frequency: usize,
//     left: usize,
//     right: usize,
// }
#[derive(Debug)]
enum Node {
    Leaf {
        frequency: usize,
        symbol: char,
    },
    Internal {
        frequency: usize,
        left: usize,
        right: usize,
    },
}

impl Node {
    pub fn new_leaf(ch: char) -> Node {
        Node::Leaf {
            symbol: ch,
            frequency: 1,
        }
    }

    pub fn new_internal(frequency: usize, left: usize, right: usize) -> Node {
        Node::Internal {
            frequency,
            left,
            right,
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            Node::Leaf { .. } => true,
            Node::Internal { .. } => false,
        }
    }
    pub fn is_internal(&self) -> bool {
        match self {
            Node::Leaf { .. } => false,
            Node::Internal { .. } => true,
        }
    }

    pub fn frequency(&self) -> usize {
        match self {
            Node::Leaf { frequency, .. } | Node::Internal { frequency, .. } => *frequency,
        }
    }
}

// '0'=>left
// '1'=>right

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct BinaryRep {
    value: u32,
    len: u8,
}

impl BinaryRep {
    pub fn new_bit(bit: u8) -> Self {
        Self {
            value: bit as u32,
            len: 1,
        }
    }

    pub fn to_new_left(&self) -> Self {
        Self {
            value: self.value << 1,
            len: self.len + 1,
        }
    }
    pub fn to_new_right(&self) -> Self {
        let mut value = self.value << 1;
        value += 1;
        Self {
            value,
            len: self.len + 1,
        }
    }

    pub fn push_bit(&self, bit: u8) -> Result<Self, HuffmanError> {
        if self.len == 32 {
            return Err(HuffmanError::IntegerOverflow);
        }

        let len = self.len + 1;
        let mut value;
        match bit {
            0 => {
                value = self.value << 1;
            }
            1 => {
                value = self.value << 1;
                value += 1;
            }
            _ => unreachable!(),
        }
        Ok(BinaryRep { value, len })
    }
}

#[derive(Error, Debug)]
pub enum HuffmanError {
    #[error("32 bit unsigned integer overflow occured while decoding")]
    IntegerOverflow,
}

// #[derive(Debug)]
// struct HuffmanIterator {
//     path: BinaryRep,
//     stack: Vec<HuffmanNode>,
// }
// impl HuffmanIterator {
//     fn next(&mut self) {}
// }

#[derive(Debug)]
struct HuffmanNode {
    index: usize,
    path: BinaryRep,
}

impl HuffmanNode {
    pub fn new_root(index: usize) -> Self {
        Self {
            index,
            path: BinaryRep { value: 0, len: 0 },
        }
    }
}

#[derive(Debug)]
enum NodeType {
    Root,
    Left,
    Right,
}

#[derive(Debug)]
struct BitWriter {
    stream: Vec<u8>,
    offset: u8,
    // for the last byte of the vector, how many bits are actually used.
    // 0 if all the bits were used.
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            stream: Vec::new(),
            offset: 0,
        }
    }
    pub fn write(&mut self, value: &BinaryRep) {
        // let mask: u32 = 0x000000ff;
        let mut remaining_bits: i8 = (value.len + self.offset) as i8;
        let remaining_space = 32 - value.len;

        // shift everything to the leftmost end minus the space for the stream's offset (if any)
        let shift_val = remaining_space - self.offset;
        let mut shifted_val = value.value << shift_val;
        //if the character takes all 4 bytes, shifting will fail as shift_val becomes -ve.
        shifted_val = shifted_val.rotate_left(8);
        let byte = shifted_val as u8;
        if self.offset > 0 {
            let last = self.stream.pop().unwrap();
            // println!("byte: {or_byte:b}", or_byte = last | byte);
            self.stream.push(last | byte);
        } else {
            self.stream.push(byte);
            // println!("byte: {byte:b}");
        }
        remaining_bits -= 8;
        while remaining_bits > 0 {
            shifted_val = shifted_val.rotate_left(8);
            let byte = shifted_val as u8;
            self.stream.push(byte);
            remaining_bits -= 8;
            // println!("byte: {byte:b}");
        }

        self.offset = (self.offset + value.len) % 8;
    }
}

// 1011 1010, 0000 1010

#[derive(Debug)]
struct BitReader<'a> {
    stream: &'a Vec<u8>,
    idx: usize,
    mask: u8,
    last_byte_len: u8, // for the last byte of the vector, how many bits are actually used.
                       // 0 if all the bits were used.
}

impl<'a> BitReader<'a> {
    pub fn new(stream: &'a Vec<u8>, last_byte_len: u8) -> Self {
        // println!("Stream: {stream:?}");
        Self {
            stream,
            mask: 0b1000_0000,
            idx: 0,
            last_byte_len: last_byte_len % 8,
        }
    }

    pub fn is_complete(&self) -> bool {
        // println!("stream reached at idx: {}", self.idx);
        if self.idx >= self.stream.len() {
            // println!("end of stream reached at idx: {}", self.idx);
            return true;
        }
        false
    }

    pub fn next_bit(&mut self) -> Option<u8> {
        let byte: u8 = match self.stream.get(self.idx) {
            Some(val) => *val,
            None => {
                println!("EOF");
                return None;
            }
        };
        if self.idx == self.stream.len() - 1 && self.last_byte_len > 0 {
            let shift_val = self.last_byte_len - 1;
            let cmp_val = 0b1000_0000 >> shift_val;
            if self.mask < cmp_val {
                return None;
            }
        }

        let bit = byte & self.mask;
        // println!("byte: {}, mask: {}", byte, self.mask);

        self.mask = self.mask.rotate_right(1);
        if self.mask == 0b1000_0000 {
            self.idx += 1;
        }
        if bit > 0 {
            // println!("\t\t => 1");
            return Some(1);
        }
        // println!("\t\t => 0");
        Some(0)
    }
}

struct HuffmanDecoder<'a> {
    code: Option<BinaryRep>,
    map: &'a HashMap<BinaryRep, char>,
    reader: BitReader<'a>,
    text: String,
}

impl<'a> HuffmanDecoder<'a> {
    pub fn with_map(
        map: &'a HashMap<BinaryRep, char>,
        stream: &'a Vec<u8>,
        last_byte_len: u8,
    ) -> Self {
        Self {
            code: None,
            reader: BitReader::new(stream, last_byte_len),
            text: String::new(),
            map,
        }
    }

    pub fn read_char(&mut self) -> anyhow::Result<Option<char>> {
        // let mut fn_count = 0;

        while !self.reader.is_complete() {
            // fn_count += 1;
            // println!("iter count: {}", fn_count);

            let bit = match self.reader.next_bit() {
                Some(b) => b,
                None => break,
            };

            // println!("before code: {:?}", self.code);

            self.code = match self.code {
                Some(value) => Some(value.push_bit(bit)?),
                None => Some(BinaryRep::new_bit(bit)),
            };

            // println!("code: {:b}", self.code.unwrap().value);

            match self.map.get(&self.code.unwrap()) {
                Some(ch) => {
                    // println!("found match for code: {:?}, match: {}", self.code, ch);
                    return Ok(Some(*ch));
                }
                None => {}
            }
        }

        Ok(None)
    }

    pub fn read(&mut self) {
        println!("\nreading stream: {:?}", self.reader.stream);
        while !self.reader.is_complete() {
            // println!(
            //     "reading index: {}, item: {:?}",
            //     self.reader.idx,
            //     self.reader.stream.get(self.reader.idx)
            // );
            let ch = match self.read_char() {
                Ok(Some(c)) => c,
                Ok(None) | Err(_) => {
                    println!("breaking at idx: {}", self.reader.idx);
                    break;
                }
            };
            self.text.push(ch);
            self.code = None;
        }
        println!(
            "last index: {}, item: {:?}",
            self.reader.idx,
            self.reader.stream.get(self.reader.idx)
        );
        println!("\n");
    }
}

fn huffman_encode(src: &str) -> Vec<u8> {
    let mut q1 = VecDeque::<Node>::new();
    let mut set = std::collections::HashMap::<char, usize>::new();
    for ch in src.chars() {
        if let Some(&idx) = set.get(&ch) {
            match q1.get_mut(idx).unwrap() {
                Node::Leaf { frequency, .. } => {
                    *frequency += 1;
                }
                Node::Internal { .. } => unreachable!(),
            };
        } else {
            _ = set.insert(ch, q1.len());
            q1.push_back(Node::new_leaf(ch));
        }
    }

    // println!("q1: {q1:?}");

    q1.make_contiguous().sort_unstable_by_key(|a| a.frequency());

    // println!("after sorting, q1: {q1:?}");

    let mut q2 = VecDeque::<Node>::new();
    let mut q3 = Vec::<Node>::new();

    // q2.iter_mut().position(|x| x.frequency>=);

    while q1.len() > 0 || q2.len() > 1 {
        //minimum of the head of the two queues
        let left = match (q1.front(), q2.front()) {
            (Some(f1), Some(f2)) => {
                if f1.frequency() <= f2.frequency() {
                    q1.pop_front()
                } else {
                    q2.pop_front()
                }
            }
            (Some(_), None) => q1.pop_front(),
            (None, Some(_)) => q2.pop_front(),
            (None, None) => break,
        }
        .unwrap();
        let right = match (q1.front(), q2.front()) {
            (Some(f1), Some(f2)) => {
                if f1.frequency() <= f2.frequency() {
                    q1.pop_front()
                } else {
                    q2.pop_front()
                }
            }
            (Some(_), None) => q1.pop_front(),
            (None, Some(_)) => q2.pop_front(),
            (None, None) => break,
        }
        .unwrap();

        let parent =
            Node::new_internal(left.frequency() + right.frequency(), q3.len(), q3.len() + 1);
        q3.push(left);
        q3.push(right);
        q2.push_back(parent);
    }
    let root = q2.pop_back().unwrap();
    q3.push(root);
    // println!("q1: {:?}", q1);
    // println!("q2: {:?}", q2);
    // println!("q3: {:?}", q3);

    let mut table = HashMap::<char, BinaryRep>::new();
    //traverse the tree from the root;
    let mut nodes = Vec::<HuffmanNode>::new();
    nodes.push(HuffmanNode::new_root(q3.len() - 1));
    println!();
    while nodes.len() > 0 {
        //pop and push the children to the stack
        let huff_node = nodes.pop().unwrap();

        let node = q3.get(huff_node.index).unwrap();

        match node {
            Node::Internal { left, right, .. } => {
                nodes.push(HuffmanNode {
                    index: *left,
                    path: huff_node.path.to_new_left(),
                });

                nodes.push(HuffmanNode {
                    index: *right,
                    path: huff_node.path.to_new_right(),
                });
            }
            Node::Leaf { symbol, .. } => {
                table.insert(*symbol, huff_node.path);
            }
        }
    }

    let mut bit_writer = BitWriter::new();
    for ch in src.chars() {
        let huffman_code = table
            .get(&ch)
            .expect("Table does not have an entry for the queried character");
        bit_writer.write(huffman_code);
    }

    // println!("Map: {:?}", table);
    // println!("text: {src}",);
    // println!(
    //     "bitwriter: {:?} with offset: {}",
    //     bit_writer.stream, bit_writer.offset
    // );
    println!("text len: {len} bits", len = (src.len() * 8));
    println!(
        "compressed len: {len} bits",
        len = (bit_writer.stream.len() * 8)
    );
    println!(
        "compression: {compression}",
        compression = (((src.len() - bit_writer.stream.len()) * 100) as f32 / src.len() as f32)
    );
    let mut output = String::new();
    output.push_str(&bit_writer.offset.to_string());
    output.push('\n');
    for (ch, v) in table {
        // println!("pushing: char= `{ch}`, val= {val:b}\n", val = v.value);

        output.push_str(&format!(
            "{ch}{val:0len$b}\n",
            val = v.value,
            len = v.len as usize
        ));
    }
    if let Some('\n') = output.chars().last() {
        output.pop().unwrap();
    }
    if let Some('\r') = output.chars().last() {
        output.pop().unwrap();
    }
    println!("stream after encoding: {:?}", bit_writer.stream);
    output.push_str("~");
    let mut bytes: Vec<u8> = output.bytes().collect();
    bytes.append(&mut bit_writer.stream);
    bytes
}

fn read_first_line(src: &[u8]) -> (usize, u8) {
    let mut iter = src.iter().enumerate();
    let mut value = String::new();
    let mut idx = 0;
    while let Some((i, ch)) = iter.next() {
        if *ch == '\n' as u8 || *ch == '\r' as u8 {
            break;
        }
        idx = i;
        value.push(*ch as char);
    }
    // println!(
    //     "idx: {idx}, value:{val}",
    //     val = value
    //         .parse::<u8>()
    //         .expect("coded string is not valid, line[1] could not be parsed")
    // );
    (
        idx + 1,
        value
            .parse()
            .expect("coded string is not valid, line[1] could not be parsed"),
    )
}
fn parse_huffman_table(src: &[u8]) -> HashMap<BinaryRep, char> {
    // let mut buf = String::new();
    // let src: Vec<u8> = src.to_vec();

    let src = String::from_utf8_lossy(src);
    // .expect("in parse_huffman_table, slice cannot be converted to string");

    let mut iter = src.lines();
    _ = iter.next();
    let mut map = HashMap::new();
    while let Some(mut line) = iter.next() {
        let mut char_iter = line.chars();
        let character = match char_iter.next() {
            Some(ch) => ch,
            None => {
                // println!("its a newline");
                line = iter.next().unwrap();
                char_iter = line.chars();
                char_iter.next().unwrap()
            }
        };
        let value: String;
        value = char_iter.take_while(|ch| *ch != '~').collect();
        // println!("[line= `{line}`] value to be parsed: {value}");
        let val = BinaryRep {
            value: u32::from_str_radix(&value, 2)
                .expect(&format!("value: {} could not be parsed from file", value)),
            len: value.len() as u8,
        };
        // println!("\t\tparsed value:{:b}", val.value);

        map.insert(val, character);
        if line.contains("~") {
            // println!("gonna return");
            return map;
        }
    }

    map
}

fn huffman_decode(src: &[u8]) -> String {
    //parse the huffmanTable
    //  2
    //  a01
    //  b10
    //  c11
    //  d001;
    // first line is the offset for last byte
    //first character=>character
    //second char till end of line => code
    let stream: Vec<u8> = src
        .bytes()
        .skip_while(|x| *x.as_ref().unwrap() != ('~' as u8))
        .skip(1)
        .map(|x| x.unwrap())
        .collect();
    println!("Stream: {stream:?}");
    let (_, last_byte_len) = read_first_line(src);
    let decode_map: HashMap<BinaryRep, char> = parse_huffman_table(src);
    // println!("decoded map: {decode_map:?}");
    let mut decoder = HuffmanDecoder::with_map(&decode_map, &stream, last_byte_len);
    decoder.read();
    decoder.text
}

fn main() {
    let timer = std::time::Instant::now();
    // println!("size of node rn= {}", std::mem::size_of::<Node>());
    // let test = "aaaabcdcdcdaaaaaaaaaa";
    // let test = "┌ ──── ┐";
    // let test = "abcd. I am abcd. efg";

    let args = env::args();
    let file_name = args.skip(1).next().expect("file name expected");
    let src = std::fs::read_to_string(file_name).unwrap();
    let encoded_src = huffman_encode(&src);
    // println!("encoded string: {:?}", encoded_src);
    let output_file_path = "encoded.txt";
    std::fs::write(output_file_path, encoded_src).expect("encoding file failed");
    let encoded = std::fs::read("encoded.txt").unwrap();
    let decoded_src = huffman_decode(&encoded);
    println!("decoded string: {}", decoded_src);

    let elapsed = timer.elapsed();
    println!();
    println!("┌─────────────────────────────┐");
    println!("│ completed in {elapsed:15.2?}│");
    println!("└─────────────────────────────┘");
}

// 2_164_864 push operations
//front: 5.92ms
//back: 3.9ms
//vec: 3.8ms
