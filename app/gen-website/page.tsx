


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
import axios from 'axios'



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

const transformFromUrl=async(e:any)=>{
      e.preventDefault();
    try {
      const res = await axios.post("http://localhost:8000/transform/url", {url: urlInput });
   
      addOutput(res.data.transformed_content);
      router.push('/output')
    } catch (error) {
      console.error(error);
    }
}

const transformFromFiles = async () => {
  setIsProcessing(true);
  setError(null);
  setSuccess(null);

  try {
    const responses = await Promise.all(
      uploadedFiles.map(async (file) => {
        const formData = new FormData();
        formData.append("file", file);

        const res = await axios.post("http://localhost:8000/transform/file", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        return res.data.transformed_content;
      })
    );

    responses.forEach((output: string) => addOutput(output));
    setSuccess(`Successfully transformed ${uploadedFiles.length} file(s)`);
    router.push("/output");
  } catch (err) {
    setError(`Error processing files: ${err instanceof Error ? err.message : "Unknown error"}`);
  } finally {
    setIsProcessing(false);
  }
};


  return (
    <div className="  h-screen bg-cover  "     style={{
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
     {/* //error */}
        </CardFooter>
      </Card>
    </div>
    </div>
  );
}

