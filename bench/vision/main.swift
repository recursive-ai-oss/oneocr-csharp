// Apple Vision text recognition over a directory of page images, for the benchmark scorer.
//
// Uses only the public Vision API (VNRecognizeTextRequest). For every page it writes
//   <id>.txt   the recognized lines in the order Vision returns them, one per line
//   <id>.json  {"words":[{"text":..,"box":[x0,y0,x1,y1]}]} in pixels of the page image, top-left origin, y down
// and, next to them, timing.json with the environment and per-page times.
//
//   vision-ocr --pages DIR --out DIR [--level accurate|fast] [--correction on|off] [--cpu-only] [--lang en-US]
//
// Word boxes come from VNRecognizedText.boundingBox(for:) on each whitespace-separated word of the top candidate.

import CoreGraphics
import Foundation
import ImageIO
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

var pages = "", out = ""
var level = "accurate", correction = false, cpuOnly = false, lang = "en-US"
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    func value() -> String {
        if args.isEmpty { fail("missing value for \(a)") }
        return args.removeFirst()
    }
    switch a {
    case "--pages": pages = value()
    case "--out": out = value()
    case "--level": level = value()
    case "--correction": correction = value() == "on"
    case "--cpu-only": cpuOnly = true
    case "--lang": lang = value()
    default: fail("unknown argument \(a)")
    }
}
if pages.isEmpty || out.isEmpty || !["accurate", "fast"].contains(level) { fail("usage: vision-ocr --pages DIR --out DIR [--level accurate|fast] [--correction on|off] [--cpu-only] [--lang en-US]") }

let fm = FileManager.default
try fm.createDirectory(atPath: out, withIntermediateDirectories: true)
let files = try fm.contentsOfDirectory(atPath: pages).filter { $0.hasSuffix(".png") || $0.hasSuffix(".jpg") }.sorted()

func peakRssMb() -> Double {
    var u = rusage()
    getrusage(RUSAGE_SELF, &u)
    return Double(u.ru_maxrss) / 1_048_576.0  // bytes on macOS
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

var perPage: [String: Double] = [:]
for file in files {
    let id = (file as NSString).deletingPathExtension
    let url = URL(fileURLWithPath: pages).appendingPathComponent(file)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        fail("cannot read \(file)")
    }
    let t0 = now()
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = level == "accurate" ? .accurate : .fast
    request.usesLanguageCorrection = correction
    request.recognitionLanguages = [lang]
    request.automaticallyDetectsLanguage = false
    request.minimumTextHeight = 0
    if cpuOnly { request.usesCPUOnly = true }  // documented (deprecated) switch: no Neural Engine or GPU
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    do { try handler.perform([request]) } catch { fail("Vision failed on \(file): \(error)") }
    perPage[id] = now() - t0

    let w = Double(image.width), h = Double(image.height)
    var lines: [String] = []
    var words: [[String: Any]] = []
    for obs in request.results ?? [] {
        guard let cand = obs.topCandidates(1).first else { continue }
        lines.append(cand.string)
        let s = cand.string
        var i = s.startIndex
        while i < s.endIndex {
            if s[i].isWhitespace { i = s.index(after: i); continue }
            var j = i
            while j < s.endIndex && !s[j].isWhitespace { j = s.index(after: j) }
            if let box = try? cand.boundingBox(for: i..<j) {
                let r = box.boundingBox  // normalized, origin bottom-left
                words.append(["text": String(s[i..<j]), "box": [r.minX * w, (1 - r.maxY) * h, r.maxX * w, (1 - r.minY) * h].map { ($0 * 10).rounded() / 10 }])
            }
            i = j
        }
    }
    try (lines.joined(separator: "\n") + "\n").write(toFile: "\(out)/\(id).txt", atomically: true, encoding: .utf8)
    let data = try JSONSerialization.data(withJSONObject: ["words": words])
    try data.write(to: URL(fileURLWithPath: "\(out)/\(id).json"))
}

let times = files.map { perPage[($0 as NSString).deletingPathExtension]! }
let info: [String: Any] = [
    "level": level, "language_correction": correction, "cpu_only": cpuOnly, "language": lang,
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "pages": files.count,
    "first_page_s": times.first ?? 0,
    "mean_s_per_page": times.isEmpty ? 0 : times.reduce(0, +) / Double(times.count),
    "mean_s_per_page_after_first": times.count > 1 ? times.dropFirst().reduce(0, +) / Double(times.count - 1) : 0,
    "peak_rss_mb": peakRssMb(),
    "per_page_s": perPage,
]
try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: "\(out)/timing.json"))
print("\(files.count) pages, level \(level), correction \(correction), cpu-only \(cpuOnly): \(String(format: "%.3f", info["mean_s_per_page"] as! Double)) s/page, peak \(String(format: "%.0f", peakRssMb())) MB")
