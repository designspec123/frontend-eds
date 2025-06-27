"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from "@/components/ui/accordion";
import { Label } from "@/components/ui/label";
import { useRouter } from "next/navigation";
import { useOutputStore } from "../useOutputStore";
import Link from "next/link";
import { ArrowBigLeft } from "lucide-react";


type ComponentRequest = {
  type: "Button" | "Radio Button" | "Dropdown" | null;
  details: {
    button_type?: string;
    num_radio?: number;
    radio_labels?: string[];
    num_dropdown?: number;
    dropdown_labels?: string[];
  };
};

export default function ComponentGenerator() {
      const reset = useOutputStore((state) => state.reset);
  
    useEffect(() => {
      reset();
    }, [reset]);

     useEffect(()=>{
        const user=localStorage.getItem('user')?JSON.parse(localStorage.getItem('user')||""):""

        if(!user?.email){
     window.location.href="/login"
}
    },[])
  const [componentRequests, setComponentRequests] = useState<ComponentRequest[]>([]);
  const [generatedComponents, setGeneratedComponents] = useState<string[]>([]);
  const router=useRouter()

  const addComponent = () => {
    setComponentRequests([...componentRequests, { type: null, details: {} }]);
  };

  const updateComponentType = (index: number, type: "Button" | "Radio Button" | "Dropdown") => {
    const updated = [...componentRequests];
    updated[index].type = type;
    updated[index].details = {};
    setComponentRequests(updated);
  };

  const updateButtonType = (index: number, buttonType: string) => {
    const updated = [...componentRequests];
    updated[index].details = { button_type: buttonType };
    setComponentRequests(updated);
  };

  const updateRadioDetails = (index: number, num: number, labels: string[]) => {
    const updated = [...componentRequests];
    updated[index].details = { num_radio: num, radio_labels: labels };
    setComponentRequests(updated);
  };

  const updateDropdownDetails = (index: number, num: number, labels: string[]) => {
    const updated = [...componentRequests];
    updated[index].details = { num_dropdown: num, dropdown_labels: labels };
    setComponentRequests(updated);
  };

  const generateComponents = () => {
    const prompts = componentRequests.map((component) => {
      if (component.type === "Button" && component.details.button_type) {
        return `Generate a ${component.details.button_type.toLowerCase()} button using Tailwind CSS.`;
      } else if (component.type === "Radio Button" && component.details.radio_labels) {
        const labels = component.details.radio_labels.join(", ");
        return `Generate a radio button group with ${component.details.num_radio} options: ${labels} using Tailwind CSS.`;
      } else if (component.type === "Dropdown" && component.details.dropdown_labels) {
        const labels = component.details.dropdown_labels.join(", ");
        return `Generate a dropdown with ${component.details.num_dropdown} options: ${labels} using Tailwind CSS.`;
      }
      return "";
    }).filter(prompt => prompt !== "");

    setGeneratedComponents(prompts);
     router.push('/output')
 
  };

  return (
    <div className="container  h-screen bg-cover text-white "     style={{
      background:"url('/images/eds-bg.jpg')"
    }}>
       <div className="pl-5 pt-5" >
            <Link href="/">
            <ArrowBigLeft className=" text-white  " /></Link>
        </div>
    <div className="mt-30 flex flex-col items-center justify-center w-full">
        <h2 className="text-2xl font-bold mb-6">Generate UI Components</h2>

      <Button onClick={addComponent} className="mb-6 backdrop-blur-md bg-white/30 text-black-500">
        Add Component
      </Button>

      <div className="space-y-4 mb-6">
        <Accordion type="multiple">
          {componentRequests.map((component, index) => (
            <AccordionItem key={index} value={`item-${index}`}>
              <AccordionTrigger>Component {index + 1}</AccordionTrigger>
              <AccordionContent>
                <div className="space-y-4">
                  <div>
                    <Label>Select Component Type</Label>
                    <Select
                      value={component.type || ""}
                      onValueChange={(value) => updateComponentType(index, value as "Button" | "Radio Button" | "Dropdown")}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a component type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Button">Button</SelectItem>
                        <SelectItem value="Radio Button">Radio Button</SelectItem>
                        <SelectItem value="Dropdown">Dropdown</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {component.type === "Button" && (
                    <div>
                      <Label>Select Button Type</Label>
                      <Select
                        value={component.details.button_type || ""}
                        onValueChange={(value) => updateButtonType(index, value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select a button type" />
                        </SelectTrigger>
                        <SelectContent>
                          {["Primary", "Secondary", "Success", "Danger", "Warning", "Info", "Light", "Dark", "Link"].map((type) => (
                            <SelectItem key={type} value={type}>{type}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {component.type === "Radio Button" && (
                    <div className="space-y-4">
                      <div>
                        <Label>Number of Radio Buttons</Label>
                        <Input
                          type="number"
                          min={1}
                          max={10}
                          className="backdrop-blur-md bg-white/30 text-black-500"
                          value={component.details.num_radio || 2}
                          onChange={(e) => {
                            const num = parseInt(e.target.value);
                            const labels = component.details.radio_labels || Array(num).fill("");
                            updateRadioDetails(index, num, labels.slice(0, num));
                          }}
                        />
                      </div>
                      <div className="space-y-2">
                        {Array.from({ length: component.details.num_radio || 2 }).map((_, i) => (
                          <div key={i}>
                            <Label>Radio Button {i + 1} Label</Label>
                            <Input
                            className="backdrop-blur-md bg-white/30 text-black-500"
                              value={component.details.radio_labels?.[i] || ""}
                              onChange={(e) => {
                                const labels = [...(component.details.radio_labels || Array(component.details.num_radio || 2).fill(""))];
                                labels[i] = e.target.value;
                                updateRadioDetails(index, component.details.num_radio || 2, labels);
                              }}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {component.type === "Dropdown" && (
                    <div className="space-y-4">
                      <div>
                        <Label>Number of Dropdown Options</Label>
                        <Input
                          type="number"
                          min={1}
                          max={10}
                          className="backdrop-blur-md bg-white/30 text-black-500"
                          value={component.details.num_dropdown || 2}
                          onChange={(e) => {
                            const num = parseInt(e.target.value);
                            const labels = component.details.dropdown_labels || Array(num).fill("");
                            updateDropdownDetails(index, num, labels.slice(0, num));
                          }}
                        />
                      </div>
                      <div className="space-y-2">
                        {Array.from({ length: component.details.num_dropdown || 2 }).map((_, i) => (
                          <div key={i}>
                            <Label>Dropdown Option {i + 1} Label</Label>
                            <Input
                            className="backdrop-blur-md bg-white/30 text-black-500"
                              value={component.details.dropdown_labels?.[i] || ""}
                              onChange={(e) => {
                                const labels = [...(component.details.dropdown_labels || Array(component.details.num_dropdown || 2).fill(""))];
                                labels[i] = e.target.value;
                                updateDropdownDetails(index, component.details.num_dropdown || 2, labels);
                              }}
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>

      <Button onClick={generateComponents} className="mb-6 backdrop-blur-md bg-white/30 text-black-500">
        Generate Components
      </Button>

      {generatedComponents.length > 0 && (
        <Card>
          <CardHeader>
            <h3 className="text-xl font-semibold">Generated Component Prompts</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {generatedComponents.map((prompt, index) => (
                <pre key={index} className="bg-gray-100 p-4 rounded-md">
                  {prompt}
                </pre>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
    </div>
  );
}