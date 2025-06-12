"use client"
import { Button } from "@/components/ui/button"
import { CopyIcon } from "lucide-react"
import { useOutputStore } from "../useOutputStore";
import { Prism } from 'react-syntax-highlighter'
import { data } from '../../lib/data'
import { okaidia } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from "react";
import { cn } from "@/lib/utils";
const OutputPage = () => {
  const [tab, setTab] = useState("preview")
  const result = useOutputStore((state => state.output))
  return <div className="container w-full p-5">
 
 <div className="bg-gray-200 rounded-sm shadow-xl pt-5 p-2 flex flex-col justify-center items-center">
     <div className="grid grid-cols-3 gap-10 mb-5 ">
      <Button className={cn("bg-cyan-500 shadow-lg flex-1 hover:bg-cyan-500 shadow-cyan-500", tab === "code" && "bg-blue-500 hover:bg-blue-500")} onClick={() => setTab('code')}>code </Button>
      <Button className={cn("bg-cyan-500 hover:bg-cyan-500 shadow-lg shadow-cyan-500", tab === "preview" && "bg-blue-500 hover:bg-blue-500")} onClick={() => setTab('preview')}>Live Preview </Button>




    </div>
 </div>
    {tab === "code" && <Prism
      langauge="html"
      showLineNumbers={true}
      wrapLines={true}
      style={okaidia}

    >
      {data}
    </Prism>}
    {tab === "preview" && <iframe
      srcDoc={data || ""}
      sandbox="allow-scripts allow-same-origin"
      className="w-full h-[800px] bg-white mt-5"
      style={{
        border: "none"
      }}

    />}
  </div>
}
export default OutputPage