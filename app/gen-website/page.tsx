


"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { useRouter } from "next/navigation";
import { useOutputStore } from "../useOutputStore";
import { ArrowBigLeft, BackpackIcon } from "lucide-react";
import Link from "next/link";




export default function WebsiteTransformer() {
    const reset = useOutputStore((state) => state.reset);
     const addOutput = useOutputStore((state) => state.addOutput);

  useEffect(() => {
    reset();
  }, [reset]);
  const [browseOption, setBrowseOption] = useState<"Browse by URL" | "Browse by File">("Browse by URL");
  const [urlInput, setUrlInput] = useState("");
 
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
      useEffect(()=>{
          const user=localStorage.getItem('user')?JSON.parse(localStorage.getItem('user')||""):""
  
          if(!user?.email){
     window.location.href="/login"
  }
      },[])
const router=useRouter()
  const handleFileUpload = (files: FileList | null) => {
    if (files) {
      setUploadedFiles(Array.from(files));
    }
 
  };

  const transformFromUrl = async () => {
    setIsProcessing(true);
    setError(null);
    setSuccess(null);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // In a real implementation, you would call your API here
      // const response = await fetch('/api/transform', {
      //   method: 'POST',
      //   body: JSON.stringify({ url: urlInput }),
      //   headers: { 'Content-Type': 'application/json' }
      // });
      // const data = await response.json();
      addOutput(`<html>
        <body>
        <p className=""bg-teal-500>hello</p>
        </body>

        </html>`)
      const filename = urlInput.split("/").pop() || "url_page.html";

      setSuccess(`Successfully transformed website from URL: ${filename}`);
        router.push('/output')
    } catch (err) {
      setError(`Error processing URL: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const transformFromFiles = async () => {
    setIsProcessing(true);
    setError(null);
    setSuccess(null);

    try {
      // Simulate API calls for each file
      await Promise.all(
        uploadedFiles.map(async (file) => {
          await new Promise(resolve => setTimeout(resolve, 1000));
          // In a real implementation:
          // const content = await file.text();
          // const response = await fetch('/api/transform', {
          //   method: 'POST',
          //   body: JSON.stringify({ content, filename: file.name }),
          //   headers: { 'Content-Type': 'application/json' }
          // });
          // return await response.json();
        })
      );

      setSuccess(`Successfully transformed ${uploadedFiles.length} file(s)`);
    } catch (err) {
      setError(`Error processing files: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="container mx-auto p-4 space-y-6 h-screen bg-cover  "     style={{
      background:"url('/images/eds-bg.jpg')"
    }}>
        <div >
            <Link href="/">
            <ArrowBigLeft className=" text-white  " /></Link>
        </div>
    <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-bold text-white text-center mt-10 mb-10">Choose how to transform a web page</h2>

      <Card className=" shadow-xl bg-pink-400/50 text-white ">
        <CardHeader>
          <div className="space-y-2">
            <Label htmlFor="browse-mode">Browse Mode</Label>
            <Select
              value={browseOption}
              onValueChange={(value) => setBrowseOption(value as "Browse by URL" | "Browse by File")}
            >
              <SelectTrigger id="browse-mode" className="w-[200px]">
                <SelectValue placeholder="Select browse mode" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Browse by URL">Browse by URL</SelectItem>
                <SelectItem value="Browse by File">Browse by File</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {browseOption === "Browse by URL" ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="url-input">Enter a full webpage URL</Label>
                <Input
                  id="url-input"
                  type="url"
                  placeholder="https://example.com"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                />
              </div>
              <Button
                onClick={transformFromUrl}
                disabled={!urlInput || isProcessing}
              >
                {isProcessing ? "Processing..." : "Transform from URL"}
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="file-upload">Upload HTML, CSS, or JS files</Label>
                <Input
                  id="file-upload"
                  type="file"
                  multiple
                  accept=".html,.css,.js"
                  onChange={(e) => handleFileUpload(e.target.files)}
                />
              </div>
              {uploadedFiles.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Selected {uploadedFiles.length} file(s):
                  </p>
                  <ul className="list-disc pl-5 text-sm">
                    {uploadedFiles.map((file, index) => (
                      <li key={index}>{file.name}</li>
                    ))}
                  </ul>
                </div>
              )}
              <Button
                onClick={transformFromFiles}
                disabled={uploadedFiles.length === 0 || isProcessing}
              >
                {isProcessing ? "Processing..." : "Transform from File(s)"}
              </Button>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex flex-col items-start gap-4">
          {/* {error && (
            <Alert variant="destructive">
              <ExclamationTriangleIcon className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {success && (
            <Alert>
              <CheckCircledIcon className="h-4 w-4" />
              <AlertTitle>Success</AlertTitle>
              <AlertDescription>{success}</AlertDescription>
            </Alert>
          )} */}
        </CardFooter>
      </Card>
    </div>
    </div>
  );
}

