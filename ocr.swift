import Foundation
import Vision
import Cocoa

// Guard that a path is provided
guard CommandLine.arguments.count > 1 else {
    print("Usage: swift ocr.swift <path_to_images_directory>")
    exit(1)
}

let dirPath = CommandLine.arguments[1]
let fm = FileManager.default

guard fm.fileExists(atPath: dirPath) else {
    print("Error: Directory not found at \(dirPath)")
    exit(1)
}

do {
    let files = try fm.contentsOfDirectory(atPath: dirPath)
    let jpegFiles = files.filter { $0.lowercased().hasSuffix(".jpeg") || $0.lowercased().hasSuffix(".jpg") }.sorted()
    
    print("Found \(jpegFiles.count) ground truth images to process.")
    
    for file in jpegFiles {
        let fullPath = (dirPath as NSString).appendingPathComponent(file)
        print("Processing: \(file)")
        
        guard let image = NSImage(contentsOfFile: fullPath),
              let tiffData = image.tiffRepresentation,
              let cgImageSource = CGImageSourceCreateWithData(tiffData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(cgImageSource, 0, nil) else {
            print("  Failed to load image: \(file)")
            continue
        }
        
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNRecognizeTextRequest { request, error in
            if let error = error {
                print("  OCR Error: \(error.localizedDescription)")
                return
            }
            
            guard let observations = request.results as? [VNRecognizedTextObservation] else {
                return
            }
            
            // Vision returns observations sorted top-to-bottom, left-to-right
            // We group them by vertical position (Y coordinate) to reconstruct the rows of the Excel sheet
            var rows: [CGFloat: [VNRecognizedTextObservation]] = [:]
            let tolerance: CGFloat = 0.015 // Vertical proximity tolerance to group items in the same line
            
            for obs in observations {
                let boundingBox = obs.boundingBox
                let y = boundingBox.origin.y
                
                // Check if we already have a row near this Y coordinate
                var matched = false
                for existingY in rows.keys {
                    if abs(existingY - y) < tolerance {
                        rows[existingY]?.append(obs)
                        matched = true
                        break
                    }
                }
                
                if !matched {
                    rows[y] = [obs]
                }
            }
            
            // Sort rows top-to-bottom (Y decreases from 1.0 to 0.0 in Vision coordinate space)
            let sortedKeys = rows.keys.sorted(by: >)
            
            for key in sortedKeys {
                if let rowObservations = rows[key] {
                    // Sort items left-to-right within the row
                    let sortedRow = rowObservations.sorted(by: { $0.boundingBox.origin.x < $1.boundingBox.origin.x })
                    let lineText = sortedRow.compactMap { $0.topCandidates(1).first?.string }.joined(separator: " | ")
                    print("  ROW: \(lineText)")
                }
            }
        }
        
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        
        try requestHandler.perform([request])
        print(String(repeating: "-", count: 50))
    }
    
} catch {
    print("Error listing directory: \(error.localizedDescription)")
}
